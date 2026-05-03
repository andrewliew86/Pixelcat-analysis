# Cat Color Quantifier 🐱🎨

Ever wondered what colors make up your cat's fur? This fun project uses image processing and machine learning to analyze cat images and extract dominant colors! Purr-fect for cat lovers and data enthusiasts alike.

## Features
- 🖼️ Load images with automatic background removal
- 🎯 Cluster pixels using K-Means algorithm
- 📊 Visualize color clusters in 3D RGB space
- 📈 Get percentages of each dominant color

## Requirements
- Python 3.x
- PIL (Pillow)
- NumPy
- Scikit-learn
- Matplotlib

## Usage
1. Place your cat images in `images/bg-removed/` (or `images/plus-bg/` for images with background)
2. Open and run `image_quantification.ipynb` in Jupyter
3. Adjust `num_colors` parameter for more/less color clusters
4. Watch the magic happen and discover your cat's color palette!

## Example Output
```
🎨 Color RGB(45, 34, 28) (#2d221c) → 45.67%
🎨 Color RGB(128, 95, 72) (#805f48) → 32.14%
🎨 Color RGB(200, 180, 150) (#c8b496) → 22.19%
```

Meow! 🐾