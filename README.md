# Cat Color Quantifier

A fun Streamlit app for cat image analysis. It can:

- find dominant fur colors
- compare color palettes between cats
- estimate how visible cat fur may be on a jacket
- score a cat's loafiness

Check out the deployed Pixel cat app for yourself on Streamlit Community Cloud: https://pixelcat-analysis.streamlit.app

## Best Inputs

Use cat images that are already cropped or segmented with a transparent background. Clean white backgrounds can also work. Busy backgrounds will make the color and loaf results less reliable.

For jacket matching, use a segmented cat image and a jacket photo where the jacket fills most of the image.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Docker

```bash
docker build -t cat-color-quantifier .
docker run --rm -p 8501:8501 cat-color-quantifier
```

## Tests

```bash
python -m pytest -q
```

## Deployment

This app needs a Python server, so it cannot run directly on GitHub Pages. Use Streamlit Community Cloud, Hugging Face Spaces, Render, Railway, or Fly.io.
