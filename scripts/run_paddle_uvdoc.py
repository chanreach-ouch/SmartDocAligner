import argparse
import sys
import os
import json

# Add parent dir to path so we can import document_unwarping
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_unwarping.paddle_uvdoc import unwarp_with_paddle

def main():
    parser = argparse.ArgumentParser(description="Run Paddle UVDoc on a single image.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--device", default="cpu", help="Device to run on (e.g., 'cpu', 'gpu:0')")
    args = parser.parse_args()
    
    print(f"Running Paddle UVDoc on {args.input}...")
    result = unwarp_with_paddle(
        image_path=args.input,
        output_path=args.output,
        device=args.device
    )
    
    print(json.dumps(result, indent=2))
    
    if result.get("error"):
        sys.exit(1)

if __name__ == "__main__":
    main()
