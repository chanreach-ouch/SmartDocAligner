# SmartDocAligner

This repository contains tools for evaluating and testing the [DocAligner](https://github.com/DocsaidLab/DocAligner) document corner detection model on the SmartDoc dataset.

## Setup

## Quick Start

1. Install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the Gradio app:
```bash
python app.py
```

*Note: On the first run, if you choose the DALAI model, the app will automatically download the ~42MB YOLOv5 pretrained weights (`model.pt`) into the project directory.*

3. Open your browser to the local URL (usually `http://127.0.0.1:7860`).

## App Features

- **DocAligner (corner detection)**: Uses the `fastvit_sa24` backbone to find the 4 corners of a document and applies a perspective transform to crop it flat.
- **Document Segmentation (DALAI)**: Uses a YOLOv5 model trained to detect document contents (text, signatures, tables, images). The app aggregates the detections and draws a bounding box to crop the document area, which is highly robust to stacked pages and occluded backgrounds.
- **Compare both**: Run both models side-by-side to evaluate which approach works best for a given image.

## Running the Evaluation Script

To run the evaluation script against the test dataset:

```bash
python evaluate_docaligner.py
```
This extracts frames, infers the corners using the `fastvit_sa24` backbone, computes metrics (pixel error and IoU), and generates visual comparisons.

## Running the Gradio Web App

We provide an interactive web app to test DocAligner with your own images (e.g. photos from your phone). To start the app locally:

```bash
python app.py
```

Then, open your web browser to `http://127.0.0.1:7860`. You can upload any paper document photo to see the detected corners and the auto-cropped result in real-time.

## Evaluation Results

Below are some examples of the model's performance on the dataset. The green box represents the ground truth, and the red box represents the model's prediction:

![Evaluation Result 1](assets/1.jpg)
![Evaluation Result 2](assets/2.webp)
![Evaluation Result 3](assets/3.webp)
