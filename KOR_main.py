import os
import numpy as np
from PIL import Image
import json
from multiprocessing import Pool, cpu_count
import time

# 🌟 전역 변수 (모든 프로세스가 공유)
tile_keys = None
tile_rgbs = None
tile_dir = None
base = 64

def init_worker(keys, rgbs, t_dir, b):
    """각 워커 프로세스 초기화"""
    global tile_keys, tile_rgbs, tile_dir, base
    tile_keys = keys
    tile_rgbs = rgbs
    tile_dir = t_dir
    base = b
    print(f"🔧 워커 프로세스 {os.getpid()} 초기화 완료")

def find_closest_images_vectorized(target_rgbs):
    """벡터화된 거리 계산"""
    global tile_rgbs, tile_keys
    
    target_rgbs = np.array(target_rgbs)
    if len(target_rgbs.shape) == 1:
        target_rgbs = target_rgbs.reshape(1, -1)
    
    # 거리 계산 (벡터화)
    distances = np.linalg.norm(
        target_rgbs[:, np.newaxis, :] - tile_rgbs[np.newaxis, :, :], 
        axis=2
    )
    closest_indices = np.argmin(distances, axis=1)
    return [tile_keys[idx] for idx in closest_indices]

def process_chunk_multiprocessing(chunk_data):
    """각 프로세스에서 실행될 함수"""
    global tile_dir, base
    
    chunk_id, img_chunk, start_y, start_x, chunk_h, chunk_w = chunk_data
    
    print(f"🔄 프로세스 {os.getpid()}: 청크 {chunk_id} 처리 시작 ({start_y},{start_x})")
    
    # 청크 내 블록들의 평균 RGB 계산
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
    
    # 벡터화된 거리 계산
    block_rgbs = [block['rgb'] for block in block_data]
    best_matches = find_closest_images_vectorized(block_rgbs)
    
    # 타일 이미지 로드 및 결과 준비
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
    
    print(f"✅ 프로세스 {os.getpid()}: 청크 {chunk_id} 완료 ({len(results)}개 블록)")
    return results

def upscale_if_too_small(img, min_size=3000):
    """이미지가 너무 작으면 확대"""
    w, h = img.size
    if w >= min_size and h >= min_size:
        return img
    
    scale = max(min_size / w, min_size / h)
    new_size = (int(w * scale), int(h * scale))
    print(f"🔁 이미지 확대: {w}x{h} → {new_size[0]}x{new_size[1]} (x{scale:.2f})")
    return img.resize(new_size, Image.BICUBIC)

def resize_to_nearest_multiple(image, base=64):
    """이미지를 base의 배수로 크기 조정"""
    w, h = image.size
    new_w = round(w / base) * base
    new_h = round(h / base) * base
    return image.resize((new_w, new_h))

def generate_mosaic_real_multiprocessing(img_path, avg_rgb_path, tile_dir_path, 
                                       output_path, base_size=64, chunk_size=512, min_size=3000):
    """진짜 멀티프로세싱 모자이크 생성"""
    
    print("🚀 멀티프로세싱 모자이크 생성 시작!")
    total_start = time.time()
    
    # 1. 이미지 로드 및 전처리
    print("📁 이미지 로드 중...")
    img = Image.open(img_path).convert('RGB')
    img = upscale_if_too_small(img, min_size)  # 너무 작으면 확대
    img = resize_to_nearest_multiple(img, base_size)  # 64의 배수로 조정
    w, h = img.size
    img_np = np.array(img)
    
    print(f"📐 이미지 크기: {w}x{h}")
    
    # 2. RGB 데이터베이스 로드
    print("📊 RGB 데이터베이스 로드 중...")
    with open(avg_rgb_path, 'r') as f:
        avg_rgb_db = json.load(f)
    
    keys = list(avg_rgb_db.keys())
    rgbs = np.array([avg_rgb_db[key] for key in keys])
    print(f"✅ {len(keys)}개 타일 데이터 로드 완료")
    
    # 3. 청크 생성
    print("🧩 청크 생성 중...")
    chunks = []
    chunk_id = 0
    
    for y in range(0, h, chunk_size):
        for x in range(0, w, chunk_size):
            chunk_h = min(chunk_size, h - y)
            chunk_w = min(chunk_size, w - x)
            img_chunk = img_np[y:y+chunk_h, x:x+chunk_w].copy()  # 복사본 생성
            
            chunks.append((chunk_id, img_chunk, y, x, chunk_h, chunk_w))
            chunk_id += 1
    
    print(f"📦 총 {len(chunks)}개 청크 생성")
    
    # 4. 멀티프로세싱 실행
    num_processes = min(cpu_count(), len(chunks))
    print(f"⚡ {num_processes}개 프로세스로 병렬 처리 시작...")
    
    processing_start = time.time()
    
    with Pool(processes=num_processes, 
              initializer=init_worker, 
              initargs=(keys, rgbs, tile_dir_path, base_size)) as pool:
        
        # 병렬 처리 실행
        all_results = pool.map(process_chunk_multiprocessing, chunks)
    
    processing_time = time.time() - processing_start
    print(f"🔥 병렬 처리 완료: {processing_time:.2f}초")
    
    # 5. 결과 조합
    print("🎨 최종 이미지 조합 중...")
    result = Image.new('RGB', (w, h))
    
    total_blocks = 0
    for chunk_results in all_results:
        for block_result in chunk_results:
            y, x = block_result['pos']
            tile_array = block_result['tile']
            tile_img = Image.fromarray(tile_array)
            result.paste(tile_img, (x, y))
            total_blocks += 1
    
    # 6. 저장
    result.save(output_path, quality=95)
    
    total_time = time.time() - total_start
    
    print(f"\n🎉 모자이크 완성!")
    print(f"📊 성능 통계:")
    print(f"   - 총 처리 시간: {total_time:.2f}초")
    print(f"   - 병렬 처리 시간: {processing_time:.2f}초")
    print(f"   - 사용된 프로세스: {num_processes}개")
    print(f"   - 처리된 청크: {len(chunks)}개")
    print(f"   - 처리된 블록: {total_blocks}개")
    print(f"   - 처리 속도: {total_blocks/total_time:.1f} 블록/초")
    print(f"   - 출력 파일: {output_path}")

# 🔍 CPU 사용률 모니터링 함수
def monitor_cpu_usage():
    """CPU 사용률 실시간 모니터링"""
    import psutil
    import threading
    import time
    
    def monitor():
        while True:
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            total_cpu = psutil.cpu_percent(interval=None)
            
            core_status = " | ".join([f"코어{i+1}: {cpu:.1f}%" 
                                    for i, cpu in enumerate(cpu_percent)])
            print(f"🖥️  CPU 사용률 - 전체: {total_cpu:.1f}% | {core_status}")
            
            if total_cpu < 10:  # CPU 사용률이 낮으면 종료
                break
            time.sleep(2)
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

# ✅ 실행 예시
if __name__ == "__main__":
    # CPU 모니터링 시작 (선택사항)
    # monitor_cpu_usage()
    
    # 멀티프로세싱 실행
    generate_mosaic_real_multiprocessing(
        img_path='input.jpg',              # 모자이크를 만들 원본 이미지 경로
        avg_rgb_path='image_avg_rgb.json',  # 타일 RGB 데이터가 담긴 JSON 파일
        tile_dir_path='Cats',               # 타일 이미지들이 있는 디렉토리
        output_path='output.jpg',  # 출력될 모자이크 이미지 경로
        base_size=64,                       # 각 타일 블록의 크기 (64x64 픽셀)
        chunk_size=512,                     # 병렬 처리를 위한 청크 크기
        min_size=5000                       # 최소 이미지 크기 (3500~5000을 추천.) 
        #가로나 세로 픽셀크기가 min_size보다 작으면 확대됨.
    )