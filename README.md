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

## Example Input
<img src="https://github.com/andrewliew86/Pixelcat-analysis/blob/main/images/bg-removed/image_1.png" width=70% height=70%>
Photo of Mochi my cat

## Example Output
<img src="https://github.com/andrewliew86/Pixelcat-analysis/blob/main/images/plots/3d-rgb-plot-example.png" width=70% height=70%>
<img src="https://github.com/andrewliew86/Pixelcat-analysis/blob/main/images/plots/donut_output.png" width=70% height=70%>
K-Means cluster plot of pixel and rough percentage quantification of color percentages
