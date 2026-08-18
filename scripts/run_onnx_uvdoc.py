import argparse
import sys
import os
import json

# Add parent dir to path so we can import document_unwarping
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_unwarping.onnx_uvdoc import unwarp_with_onnx

def main():
    parser = argparse.ArgumentParser(description="Run ONNX UVDoc on a single image.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--model", default="models/UVDoc_infer.onnx", help="Path to ONNX model")
    parser.add_argument("--providers", nargs="+", help="ONNX execution providers (e.g., CUDAExecutionProvider CPUExecutionProvider)")
    args = parser.parse_args()
    
    print(f"Running ONNX UVDoc on {args.input} using model {args.model}...")
    result = unwarp_with_onnx(
        image_path=args.input,
        output_path=args.output,
        model_path=args.model,
        providers=args.providers
    )
    
    print(json.dumps(result, indent=2))
    
    if result.get("error"):
        sys.exit(1)

if __name__ == "__main__":
    main()
