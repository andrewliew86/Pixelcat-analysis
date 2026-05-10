<div align='center'>

![Pixelcat Logo](https://github.com/andrewliew86/Pixelcat-analysis/blob/main/images/plots/pixel_cat_logo.png?raw=true)

# 🎨 Cat Color Quantifier 🐱
</div>

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

### Notebook
1. Git clone repository
1. Place your cat images in `images/bg-removed/` (or `images/plus-bg/` for images with background)
2. Open and run `image_quantification.ipynb` in Jupyter
3. Adjust `num_colors` parameter for more/less color clusters
4. Watch the magic happen and discover your cat's color palette!

### Streamlit app
A simple web UI is also provided in `app.py`.

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Or with Docker:
```bash
docker build -t cat-color-quantifier .
docker run --rm -p 8501:8501 cat-color-quantifier
```
Then open http://localhost:8501, upload an image, and pick the number of color clusters.

> **Note on backgrounds:** for best results, upload an image with the background already removed (e.g. a transparent-background PNG). If the uploaded image has no alpha channel, the app falls back to a simple heuristic that treats near-white pixels (R, G, B all > 200) as background and ignores them. This works for cats photographed against a clean white backdrop but will misclassify white fur or busy/dark backgrounds — pre-remove the background for anything else.


## Example output from streamlit app
<img src="https://github.com/andrewliew86/Pixelcat-analysis/blob/main/images/plots/donut_output.png" width=70% height=70%>
Simple streamlit app showing input image, quantitative output (donut plot)  
<img src="https://github.com/andrewliew86/Pixelcat-analysis/blob/main/images/plots/3d-rgb-plot-example.png" width=70% height=70%>
K-Means cluster plot of pixel and rough percentage quantification of color percentages
