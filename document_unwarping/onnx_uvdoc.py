import os
import cv2
import numpy as np
from .common import format_result, Timer

_onnx_sessions = {}

def get_onnx_session(model_path: str, providers: list[str] = None):
    global _onnx_sessions
    key = (model_path, tuple(providers) if providers else None)
    if key not in _onnx_sessions:
        import onnxruntime as ort
        if providers is None:
            providers = ["CPUExecutionProvider"]
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _onnx_sessions[key] = ort.InferenceSession(model_path, providers=providers)
    return _onnx_sessions[key]

def unwarp_with_onnx(
    image_path: str,
    output_path: str,
    model_path: str = "models/UVDoc_infer.onnx",
    providers: list[str] = None,
) -> dict:
    try:
        if not os.path.exists(image_path):
            return format_result(image_path, output_path, 0, 0, 0.0, "ONNX", error_msg=f"Input file not found: {image_path}")
        if not os.path.exists(model_path):
            return format_result(image_path, output_path, 0, 0, 0.0, "ONNX", error_msg=f"Model not found: {model_path}")
            
        session = get_onnx_session(model_path, providers)
        
        # Determine actual providers used
        actual_providers = session.get_providers()
        
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        
        input_name = input_meta.name
        output_name = output_meta.name
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return format_result(image_path, output_path, 0, 0, 0.0, "ONNX", error_msg=f"Failed to read image: {image_path}")
            
        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Inspect expected input shape from ONNX model
        # shape is usually [batch, channels, height, width] or [batch, height, width, channels]
        in_shape = input_meta.shape
        
        # Assuming NCHW standard for PyTorch/Paddle exported models
        if len(in_shape) == 4 and in_shape[1] == 3:
            target_h, target_w = in_shape[2], in_shape[3]
            # Handle dynamic shapes
            if type(target_h) == str or target_h is None: target_h = 512
            if type(target_w) == str or target_w is None: target_w = 512
            
            resized = cv2.resize(img_rgb, (target_w, target_h))
            # Standard normalization
            # Note: UVDoc usually normalizes to [0, 1] then applies mean/std
            resized_float = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (resized_float - mean) / std
            
            # HWC to NCHW
            input_tensor = np.transpose(normalized, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
        else:
            raise ValueError(f"Unsupported input shape: {in_shape}")

        with Timer() as t:
            outputs = session.run([output_name], {input_name: input_tensor})
            
        out_tensor = outputs[0]
        out_shape = out_tensor.shape
        
        # Post-process based on output shape
        # If output is [1, 2, H, W] -> it's a UV grid
        # If output is [1, 3, H, W] -> it's an image
        if len(out_shape) == 4 and out_shape[1] == 2:
            # UV Grid Remapping
            grid = out_tensor[0] # [2, H, W]
            grid = np.transpose(grid, (1, 2, 0)) # [H, W, 2]
            
            # The grid usually contains coordinates in [-1, 1] or absolute coords
            # Assuming [-1, 1] normalized coordinates:
            if grid.min() >= -1.5 and grid.max() <= 1.5:
                grid_x = (grid[..., 0] + 1) * orig_w / 2.0
                grid_y = (grid[..., 1] + 1) * orig_h / 2.0
            else:
                # Absolute coordinates or relative [0, 1]
                if grid.max() <= 1.5:
                    grid_x = grid[..., 0] * orig_w
                    grid_y = grid[..., 1] * orig_h
                else:
                    grid_x = grid[..., 0]
                    grid_y = grid[..., 1]
                    
            map_x = grid_x.astype(np.float32)
            map_y = grid_y.astype(np.float32)
            
            unwarped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        elif len(out_shape) == 4 and out_shape[1] == 3:
            # Direct image output
            img_out = out_tensor[0] # [3, H, W]
            img_out = np.transpose(img_out, (1, 2, 0))
            # Denormalize
            img_out = (img_out * std + mean) * 255.0
            img_out = np.clip(img_out, 0, 255).astype(np.uint8)
            unwarped = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unknown output shape format: {out_shape}")
            
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, unwarped)
        
        final_h, final_w = unwarped.shape[:2]
        
        return format_result(
            input_path=image_path,
            output_path=output_path,
            model_path=model_path,
            providers=actual_providers,
            input_shape=list(in_shape),
            output_shape=list(out_shape),
            width=final_w,
            height=final_h,
            inference_time_ms=t.elapsed_ms,
            runtime_name="ONNX"
        )
    except Exception as e:
        import traceback
        return format_result(
            image_path, output_path, 0, 0, 0.0, "ONNX", 
            error_msg=str(e) + "\n" + traceback.format_exc()
        )
