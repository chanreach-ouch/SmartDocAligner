# UVDoc Document Unwarping

This module provides document unwarping capabilities using the UVDoc deep learning model. Two backends are supported: PaddleOCR and ONNX.

## ⚠️ Important Warning
UVDoc performs **unwarping/rectification** of a document image that is already cropped or closely bounded. It is **not** guaranteed to detect the document boundary in a cluttered full-camera image. The input to these models should be an image that already contains the document, or an image that was previously cropped by OpenCV/DocAligner.

## Backends

### 1. PaddleOCR Backend
- **Source:** PaddlePaddle/PaddleOCR
- **License:** Apache 2.0
- **Usage:** Automatically downloads the UVDoc model weights if not found locally.
- **Run:** `python scripts/run_paddle_uvdoc.py --input <input> --output <output>`

### 2. ONNX Backend
- **Usage:** Requires a local `.onnx` file (e.g., `models/UVDoc_infer.onnx`).
- **Hardware:** Uses ONNX Runtime. Falls back to CPU if CUDA is not available. To use GPU, you must install `onnxruntime-gpu` explicitly (`pip install onnxruntime-gpu`).
- **Run:** `python scripts/run_onnx_uvdoc.py --input <input> --output <output> --model <path_to_onnx>`

## Integration

You can easily select the backend in your code:

```python
backend = "paddle" # or "onnx"
if backend == "paddle":
    from document_unwarping.paddle_uvdoc import unwarp_with_paddle
    res = unwarp_with_paddle(image_path, output_path)
else:
    from document_unwarping.onnx_uvdoc import unwarp_with_onnx
    res = unwarp_with_onnx(image_path, output_path, model_path="models/UVDoc_infer.onnx")
```

Both backends return a structured dictionary containing inference statistics and error messages (if any). The original image is not silently overwritten on failure.

## Testing

Run tests with `pytest tests/test_unwarping.py`. Tests use mocked sessions so they do not require downloading heavy model weights.

## Utilities

- **Inspect ONNX model:** `python scripts/inspect_uvdoc_onnx.py --model models/UVDoc_infer.onnx`
- **Compare both backends:** `python scripts/compare_uvdoc.py --input <input> --paddle-output <out_p> --onnx-output <out_o>`
