import os
import numpy as np
from PIL import Image
import json

def compute_dataset_mean_colors(data_dir='Cats', output_file='image_avg_rgb.json', prefix='name'):
    """
    Compute average RGB values for all images in dataset
    
    Args:
        data_dir (str): Directory containing images
        output_file (str): Output JSON file path
        prefix (str): Filename prefix to filter images
    """
    result = {}
    filenames = sorted([
        fname for fname in os.listdir(data_dir)
        if fname.lower().endswith('.png') and fname.startswith(prefix)
    ])

    print(f"🔍 Processing {len(filenames)} images with prefix '{prefix}'...")
    
    for i, filename in enumerate(filenames, 1):
        path = os.path.join(data_dir, filename)
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        mean_color = img_np.mean(axis=(0, 1)).tolist()  # [R, G, B]
        
        key = os.path.splitext(filename)[0]
        result[key] = mean_color
        print(f"📷 [{i:4d}/{len(filenames)}] {key}: RGB{mean_color}")
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Completed: Processed {len(result)} files, results saved to {output_file}")
    return result

# ✅ Execute
if __name__ == "__main__":
    compute_dataset_mean_colors(data_dir='Cats', output_file='image_avg_rgb.json', prefix='cat_')
    # Parameters:
    # - data_dir: Directory containing the image dataset (default: 'Cats')
    # - output_file: JSON file path to save the results (default: 'image_avg_rgb.json')
    # - prefix: Filename prefix to filter specific images
    # example: cat_0.png, cat_1.png, cat_2.png, ...
    # example: name0.png, name1.png, name2.png, ...
    