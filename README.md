! [Demo Image] (./Demo.jpg)

# Photo Mosaic Generator 🎨

Transform any image into a stunning photo mosaic by replacing each pixel block with another image! This project uses multiprocessing for fast generation and creates beautiful artistic effects.

## What is Photo Mosaic?

Photo mosaic (also called photomosaic) is an artistic technique where a large image is composed of many smaller images. Each small image acts as a "tile" that represents a pixel or block of pixels in the original image, creating a unique visual effect when viewed from a distance.

## Features ✨

- **High Performance**: Utilizes multiprocessing for fast mosaic generation
- **Customizable Tile Size**: Adjust the size of individual mosaic tiles
- **Automatic Image Preprocessing**: Smart upscaling and resizing
- **Real-time Progress Monitoring**: Track processing progress and CPU usage
- **Flexible Tile Sources**: Use any collection of square images as tiles

## Default Example: Cat Mosaic 🐱

By default, this project transforms your photos into **cat mosaics** - replacing every pixel block with adorable cat photos! The result is a beautiful artwork where your original image is recreated entirely using cat pictures.

## Quick Start 🚀

### Prerequisites

```bash
pip install numpy pillow psutil
```

### Basic Usage

1. **Place your input image** in the project folder (name it `input2.jpg` or modify the path in code)

2. **Navigate to project directory**
   ```bash
   cd photo-mosaic-generator
   ```

3. **Run the mosaic generator**
   ```bash
   # For Korean interface
   python KOR_main.py
   
   # For English interface  
   python EN_main.py
   ```

4. **Check your results!** The output will be saved as `mosaic_multiprocessing1.jpg`

### Adjusting Cat Size

If the cats appear too small or too large in your mosaic, modify the `min_size` parameter:

```python
generate_mosaic_real_multiprocessing(
    # ... other parameters ...
    min_size=5000  # Increase for larger cats, decrease for smaller cats
)
```

## Using Custom Images 🖼️

Want to create mosaics with your own images instead of cats? Follow these steps:

### Step 1: Prepare Your Tile Images

1. **Create a new folder** for your tile images
2. **Add your images** to this folder (square images work best!)
3. **Use diverse images** for better mosaic quality
4. **Ensure consistent naming** (e.g., `dog_001.png`, `dog_002.png`, etc.)

### Step 2: Generate Image Database

1. **Modify the preprocessing script parameters:**
   ```python
   compute_dataset_mean_colors(
       data_dir='YourImageFolder',    # Your new folder name
       output_file='custom_avg_rgb.json',
       prefix='dog_'                  # Your image prefix
   )
   ```

2. **Run preprocessing:**
   ```bash
   python imgtojson.py
   ```

### Step 3: Update Main Script

Modify the main script parameters:

```python
generate_mosaic_real_multiprocessing(
    img_path='input.jpg',
    avg_rgb_path='custom_avg_rgb.json',      # Your JSON file
    tile_dir_path='YourImageFolder',         # Your image folder
    output_path='output.jpg',
    base_size=64,                            # Match your tile image size
    chunk_size=512,
    min_size=5000
)
```

### Step 4: Generate Your Custom Mosaic

```bash
python EN_main.py  # or KOR_main.py
```

## Parameters Explained 🔧

| Parameter | Description | Default |
|-----------|-------------|---------|
| `img_path` | Input image file path | `'input.jpg'` |
| `avg_rgb_path` | JSON file with tile RGB data | `'image_avg_rgb.json'` |
| `tile_dir_path` | Directory containing tile images | `'Data'` |
| `output_path` | Output mosaic file path | `'output.jpg'` |
| `base_size` | Size of each mosaic tile in pixels | `64` |
| `chunk_size` | Chunk size for multiprocessing | `512` |
| `min_size` | Minimum image size (auto-upscaled if smaller) | `3000` |

## File Structure 📁

```
photo-mosaic-generator/
├── EN_main.py              # English version main script
├── KOR_main.py             # Korean version main script  
├── imgtojson.py            # Image preprocessing script
├── Data/                   # Default cat images folder
│   ├── cat_0.png
│   ├── cat_1.png
│   └── ...
├── image_avg_rgb.json      # Preprocessed RGB data
├── input.jpg              # Your input image
└── README.md
```

## Performance Tips 💡

- **Image Quality**: Use high-quality, diverse tile images for better results
- **Processing Speed**: The script automatically uses all available CPU cores
- **Memory Usage**: Large images may require significant RAM
- **Tile Diversity**: More diverse tiles create more detailed mosaics

## Troubleshooting 🔧

**Issue**: Mosaic looks blurry or low quality
- **Solution**: Increase `min_size` parameter or use higher resolution input image

**Issue**: Process runs out of memory
- **Solution**: Reduce `chunk_size` or use smaller input image

**Issue**: Colors don't match well
- **Solution**: Use more diverse tile images or ensure good color coverage

## License 📄

This project is open source. Feel free to modify and distribute!

## Contributing 🤝

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Credits 🙏

- **Cat Dataset**: Cat Dataset from Kaggle - https://www.kaggle.com/datasets/borhanitrash/cat-dataset
---

**Created with ❤️ for photo mosaic enthusiasts!**