import os
import numpy as np
from PIL import Image
import json
from multiprocessing import Pool, cpu_count
import time

# 🌟 Global variables (shared across all processes)
tile_keys = None
tile_rgbs = None
tile_dir = None
base = 64

def init_worker(keys, rgbs, t_dir, b):
    """Initialize each worker process"""
    global tile_keys, tile_rgbs, tile_dir, base
    tile_keys = keys
    tile_rgbs = rgbs
    tile_dir = t_dir
    base = b
    print(f"🔧 Worker process {os.getpid()} initialized")

def find_closest_images_vectorized(target_rgbs):
    """Vectorized distance calculation"""
    global tile_rgbs, tile_keys
    
    target_rgbs = np.array(target_rgbs)
    if len(target_rgbs.shape) == 1:
        target_rgbs = target_rgbs.reshape(1, -1)
    
    # Distance calculation (vectorized)
    distances = np.linalg.norm(
        target_rgbs[:, np.newaxis, :] - tile_rgbs[np.newaxis, :, :], 
        axis=2
    )
    closest_indices = np.argmin(distances, axis=1)
    return [tile_keys[idx] for idx in closest_indices]

def process_chunk_multiprocessing(chunk_data):
    """Function to be executed in each process"""
    global tile_dir, base
    
    chunk_id, img_chunk, start_y, start_x, chunk_h, chunk_w = chunk_data
    
    print(f"🔄 Process {os.getpid()}: Processing chunk {chunk_id} ({start_y},{start_x})")
    
    # Calculate average RGB for blocks within the chunk
    block_data = []
    
    for y in range(0, chunk_h, base):
        for x in range(0, chunk_w, base):
            if y + base <= chunk_h and x + base <= chunk_w:
                block = img_chunk[y:y+base, x:x+base]
                block_mean = block.mean(axis=(0,1))
                block_data.append({
                    'rgb': block_mean,
                    'pos': (start_y + y, start_x + x)
                })
    
    if not block_data:
        return []
    
    # Vectorized distance calculation
    block_rgbs = [block['rgb'] for block in block_data]
    best_matches = find_closest_images_vectorized(block_rgbs)
    
    # Load tile images and prepare results
    results = []
    for i, block in enumerate(block_data):
        tile_name = best_matches[i]
        tile_path = os.path.join(tile_dir, tile_name + '.png')
        
        if os.path.exists(tile_path):
            tile_img = Image.open(tile_path).resize((base, base))
            results.append({
                'pos': block['pos'],
                'tile': np.array(tile_img),
                'tile_name': tile_name
            })
    
    print(f"✅ Process {os.getpid()}: Chunk {chunk_id} completed ({len(results)} blocks)")
    return results

def upscale_if_too_small(img, min_size=3000):
    """Upscale image if it's too small"""
    w, h = img.size
    if w >= min_size and h >= min_size:
        return img
    
    scale = max(min_size / w, min_size / h)
    new_size = (int(w * scale), int(h * scale))
    print(f"🔁 Upscaling image: {w}x{h} → {new_size[0]}x{new_size[1]} (x{scale:.2f})")
    return img.resize(new_size, Image.BICUBIC)

def resize_to_nearest_multiple(image, base=64):
    """Resize image to nearest multiple of base"""
    w, h = image.size
    new_w = round(w / base) * base
    new_h = round(h / base) * base
    return image.resize((new_w, new_h))

def generate_mosaic_real_multiprocessing(img_path, avg_rgb_path, tile_dir_path, 
                                       output_path, base_size=64, chunk_size=512, min_size=3000):
    """Generate mosaic using real multiprocessing"""
    
    print("🚀 Starting multiprocessing mosaic generation!")
    total_start = time.time()
    
    # 1. Load and preprocess image
    print("📁 Loading image...")
    img = Image.open(img_path).convert('RGB')
    img = upscale_if_too_small(img, min_size)  # Upscale if too small
    img = resize_to_nearest_multiple(img, base_size)  # Adjust to multiple of 64
    w, h = img.size
    img_np = np.array(img)
    
    print(f"📐 Image size: {w}x{h}")
    
    # 2. Load RGB database
    print("📊 Loading RGB database...")
    with open(avg_rgb_path, 'r') as f:
        avg_rgb_db = json.load(f)
    
    keys = list(avg_rgb_db.keys())
    rgbs = np.array([avg_rgb_db[key] for key in keys])
    print(f"✅ Loaded {len(keys)} tile data successfully")
    
    # 3. Create chunks
    print("🧩 Creating chunks...")
    chunks = []
    chunk_id = 0
    
    for y in range(0, h, chunk_size):
        for x in range(0, w, chunk_size):
            chunk_h = min(chunk_size, h - y)
            chunk_w = min(chunk_size, w - x)
            img_chunk = img_np[y:y+chunk_h, x:x+chunk_w].copy()  # Create copy
            
            chunks.append((chunk_id, img_chunk, y, x, chunk_h, chunk_w))
            chunk_id += 1
    
    print(f"📦 Created {len(chunks)} chunks in total")
    
    # 4. Execute multiprocessing
    num_processes = min(cpu_count(), len(chunks))
    print(f"⚡ Starting parallel processing with {num_processes} processes...")
    
    processing_start = time.time()
    
    with Pool(processes=num_processes, 
              initializer=init_worker, 
              initargs=(keys, rgbs, tile_dir_path, base_size)) as pool:
        
        # Execute parallel processing
        all_results = pool.map(process_chunk_multiprocessing, chunks)
    
    processing_time = time.time() - processing_start
    print(f"🔥 Parallel processing completed: {processing_time:.2f}s")
    
    # 5. Combine results
    print("🎨 Combining final image...")
    result = Image.new('RGB', (w, h))
    
    total_blocks = 0
    for chunk_results in all_results:
        for block_result in chunk_results:
            y, x = block_result['pos']
            tile_array = block_result['tile']
            tile_img = Image.fromarray(tile_array)
            result.paste(tile_img, (x, y))
            total_blocks += 1
    
    # 6. Save
    result.save(output_path, quality=95)
    
    total_time = time.time() - total_start
    
    print(f"\n🎉 Mosaic generation completed!")
    print(f"📊 Performance Statistics:")
    print(f"   - Total processing time: {total_time:.2f}s")
    print(f"   - Parallel processing time: {processing_time:.2f}s")
    print(f"   - Processes used: {num_processes}")
    print(f"   - Chunks processed: {len(chunks)}")
    print(f"   - Blocks processed: {total_blocks}")
    print(f"   - Processing speed: {total_blocks/total_time:.1f} blocks/s")
    print(f"   - Output file: {output_path}")

# 🔍 CPU usage monitoring function
def monitor_cpu_usage():
    """Real-time CPU usage monitoring"""
    import psutil
    import threading
    import time
    
    def monitor():
        while True:
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            total_cpu = psutil.cpu_percent(interval=None)
            
            core_status = " | ".join([f"Core{i+1}: {cpu:.1f}%" 
                                    for i, cpu in enumerate(cpu_percent)])
            print(f"🖥️  CPU Usage - Total: {total_cpu:.1f}% | {core_status}")
            
            if total_cpu < 10:  # Exit if CPU usage is low
                break
            time.sleep(2)
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

# ✅ Usage example
if __name__ == "__main__":
    # Start CPU monitoring (optional)
    # monitor_cpu_usage()
    
    # Execute multiprocessing
    generate_mosaic_real_multiprocessing(
        img_path='input.jpg',              # Input image path to create mosaic from
        avg_rgb_path='image_avg_rgb.json',  # JSON file containing tile RGB data
        tile_dir_path='Cats',               # Directory containing tile images (Default: Cats)
        output_path='output.jpg',  # Output mosaic image path
        base_size=64,                       # Size of each tile block (64x64 pixels)
        chunk_size=512,                     # Size of chunks for parallel processing
        min_size=5000                       # Minimum image size (Recommended: 3500~5000)
        # if height or width of size is smaller than minsize, it will be upscaled.
    )