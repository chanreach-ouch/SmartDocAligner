import argparse
import sys
import os
import json
import cv2
import numpy as np

# Add parent dir to path so we can import document_unwarping
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_unwarping.paddle_uvdoc import unwarp_with_paddle
from document_unwarping.onnx_uvdoc import unwarp_with_onnx

def main():
    parser = argparse.ArgumentParser(description="Compare Paddle and ONNX UVDoc implementations.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--paddle-output", required=True, help="Output image path for Paddle")
    parser.add_argument("--onnx-output", required=True, help="Output image path for ONNX")
    parser.add_argument("--onnx-model", default="models/UVDoc_infer.onnx", help="Path to ONNX model")
    args = parser.parse_args()
    
    print(f"Comparing UVDoc implementations on {args.input}...")
    
    # Run Paddle
    print("\nRunning PaddleOCR UVDoc...")
    paddle_res = unwarp_with_paddle(args.input, args.paddle_output)
    if paddle_res.get("error"):
        print(f"PaddleOCR Failed: {paddle_res['error']}")
        
    # Run ONNX
    print("\nRunning ONNX UVDoc...")
    onnx_res = unwarp_with_onnx(args.input, args.onnx_output, args.onnx_model)
    if onnx_res.get("error"):
        print(f"ONNX Failed: {onnx_res['error']}")
        
    print("\n--- Comparison Report ---")
    print(f"Paddle output path: {paddle_res['output_path']}")
    print(f"ONNX output path: {onnx_res['output_path']}")
    print(f"Paddle inference time: {paddle_res.get('inference_time_ms', 0):.2f} ms")
    print(f"ONNX inference time: {onnx_res.get('inference_time_ms', 0):.2f} ms")
    print(f"Selected ONNX providers: {onnx_res.get('providers', [])}")
    
    p_w, p_h = paddle_res.get('width', 0), paddle_res.get('height', 0)
    o_w, o_h = onnx_res.get('width', 0), onnx_res.get('height', 0)
    
    print(f"Paddle Output Dimensions: {p_w}x{p_h}")
    print(f"ONNX Output Dimensions: {o_w}x{o_h}")
    
    failed = False
    if paddle_res.get("error") or onnx_res.get("error"):
        print("\nStatus: One or both implementations FAILED.")
        failed = True
    else:
        # Compute mean absolute pixel difference if dimensions match
        if p_w == o_w and p_h == o_h and p_w > 0:
            img_p = cv2.imread(args.paddle_output)
            img_o = cv2.imread(args.onnx_output)
            if img_p is not None and img_o is not None:
                diff = np.abs(img_p.astype(np.float32) - img_o.astype(np.float32))
                mean_diff = np.mean(diff)
                print(f"Mean absolute pixel difference: {mean_diff:.4f}")
                if mean_diff > 10.0:
                    print("Warning: Outputs differ significantly!")
            else:
                print("Failed to read output images for comparison.")
        else:
            print("Mean absolute pixel difference: N/A (dimensions mismatch)")

    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
