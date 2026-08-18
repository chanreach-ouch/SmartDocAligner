import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Inspect ONNX model inputs and outputs.")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    args = parser.parse_args()
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed. Please install it using 'pip install onnxruntime'.")
        sys.exit(1)
        
    try:
        session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        sys.exit(1)
        
    print(f"Model: {args.model}")
    print("-" * 50)
    
    print("Inputs:")
    for i, meta in enumerate(session.get_inputs()):
        print(f"  Input {i}:")
        print(f"    Name: {meta.name}")
        print(f"    Shape: {meta.shape}")
        print(f"    Type: {meta.type}")
        
    print("\nOutputs:")
    for i, meta in enumerate(session.get_outputs()):
        print(f"  Output {i}:")
        print(f"    Name: {meta.name}")
        print(f"    Shape: {meta.shape}")
        print(f"    Type: {meta.type}")
        
    print("\nProviders:")
    available = ort.get_available_providers()
    print(f"  Available: {available}")
    print(f"  Selected: {session.get_providers()}")

if __name__ == "__main__":
    main()
