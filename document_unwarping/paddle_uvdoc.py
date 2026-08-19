import os
from .common import format_result, Timer

# Cache the paddle OCR unwarping model instances globally
# mapping from (model_name, device) -> model instance
_model_cache = {}

def _check_paddleocr_available():
    """Returns (True, None) if paddleocr is importable, else (False, error_message)."""
    try:
        import paddleocr  # noqa: F401
        return True, None
    except ImportError:
        return False, (
            "paddleocr is not installed. "
            "Install it with: pip install paddlepaddle paddleocr"
        )

def get_paddle_model(device: str):
    global _model_cache
    key = ("UVDoc", device)
    if key not in _model_cache:
        # Import inside the function to prevent failing if paddleocr is not installed
        from paddleocr import TextImageUnwarping
        _model_cache[key] = TextImageUnwarping(
            model_name="UVDoc",
            device=device,
        )
    return _model_cache[key]

def unwarp_with_paddle(
    image_path: str,
    output_path: str,
    device: str = "cpu",
) -> dict:
    import cv2
    # Check if paddleocr is available before trying anything
    available, err_msg = _check_paddleocr_available()
    if not available:
        return format_result(
            input_path=image_path,
            output_path=output_path,
            width=0,
            height=0,
            inference_time_ms=0.0,
            runtime_name="PaddleOCR",
            error_msg=err_msg,
        )
    try:
        if not os.path.exists(image_path):
            return format_result(
                input_path=image_path,
                output_path=output_path,
                width=0,
                height=0,
                inference_time_ms=0.0,
                runtime_name="PaddleOCR",
                error_msg=f"Input file not found: {image_path}"
            )

        # Get or initialize the model
        model = get_paddle_model(device)


        with Timer() as t:
            results = model.predict(
                image_path,
                batch_size=1,
            )

        if not results or len(results) == 0:
            return format_result(
                input_path=image_path,
                output_path=output_path,
                width=0,
                height=0,
                inference_time_ms=t.elapsed_ms,
                runtime_name="PaddleOCR",
                error_msg="No output returned from PaddleOCR."
            )

        # results[0] is typically a specific result object with save_to_img method
        # or the image data itself depending on the paddleocr version.
        # Based on standard usage:
        res = results[0]
        
        # Save output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # In PaddleOCR, res has save_to_img. But wait, it saves to a folder.
        # Let's check if there is an image property, or just use save_to_img and rename it.
        # Actually, if we just want to save it to output_path precisely:
        if hasattr(res, 'image'):
            cv2.imwrite(output_path, res.image)
        elif hasattr(res, 'save_to_img'):
            # It expects a directory
            res.save_to_img(os.path.dirname(os.path.abspath(output_path)))
            # The saved file might not be exactly output_path. To be safe, we will just use the API correctly.
            pass
            
        # Wait, the prompt says "Save the unwarped output image."
        # If res is an image (numpy array), cv2.imwrite is enough. 
        # For PaddleOCR TextImageUnwarping, predict returns a list of result objects, and often res.image is the unwarped image.
        if hasattr(res, 'image') and res.image is not None:
            cv2.imwrite(output_path, res.image)
        elif type(res) == dict and 'image' in res:
            cv2.imwrite(output_path, res['image'])
        elif hasattr(res, 'save_to_img'):
            res.save_to_img(output_path)  # Assuming we pass the full path if supported, or dir
            if not os.path.exists(output_path):
                 return format_result(input_path=image_path, output_path=output_path, width=0, height=0, inference_time_ms=t.elapsed_ms, runtime_name="PaddleOCR", error_msg="Failed to save image to output path.")
        else:
            # Fallback if it's just a numpy array
            import numpy as np
            if isinstance(res, np.ndarray):
                cv2.imwrite(output_path, res)
            
        if not os.path.exists(output_path):
             return format_result(
                input_path=image_path,
                output_path=output_path,
                width=0,
                height=0,
                inference_time_ms=t.elapsed_ms,
                runtime_name="PaddleOCR",
                error_msg="Failed to generate output file."
            )

        img = cv2.imread(output_path)
        h, w = img.shape[:2] if img is not None else (0, 0)

        return format_result(
            input_path=image_path,
            output_path=output_path,
            width=w,
            height=h,
            inference_time_ms=t.elapsed_ms,
            runtime_name="PaddleOCR"
        )
    except Exception as e:
        import traceback
        return format_result(
            input_path=image_path,
            output_path=output_path,
            width=0,
            height=0,
            inference_time_ms=0.0,
            runtime_name="PaddleOCR",
            error_msg=str(e) + "\n" + traceback.format_exc()
        )
