import torch
import cv2
import numpy as np
import os

model = None

def load_dalai_model():
    global model
    if model is not None:
        return True
        
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
    if not os.path.exists(model_path):
        print(f"Error: DALAI model weights not found at {model_path}.")
        return False
        
    print("Initializing DALAI segmentation model...")
    try:
        # Prefer the `yolov5` pip package — it loads locally without network access.
        import yolov5
        model = yolov5.load(model_path)
        model.conf = 0.35  # Recommended in DALAI README
        model.names = {0:'typewritten', 1:'handwritten', 2:'signature', 3:'image', 4:'table'}
        print("DALAI model loaded successfully (via yolov5 package).")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"yolov5 package load failed ({e}), trying torch.hub...")

    try:
        # Fallback: torch.hub (requires network on first run)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False, trust_repo=True)
        model.conf = 0.35
        model.names = {0:'typewritten', 1:'handwritten', 2:'signature', 3:'image', 4:'table'}
        print("DALAI model loaded successfully (via torch.hub).")
        return True
    except Exception as e:
        print(f"Error loading DALAI model: {e}")
        return False


def detect_document_segmentation(image):
    """
    Runs the DALAI YOLOv5 model on the image, aggregates bounding boxes of 
    all detected contents (text, tables, images, signatures) and derives a 
    bounding box that encompasses the document contents.
    
    Returns:
        vis_img: image with overlaid masks/bounding boxes.
        cropped_img: cropped image representing the document.
        error_msg: status or error message.
    """
    if not load_dalai_model():
        return image, None, "❌ Failed to load DALAI model. Please ensure 'model.pt' is downloaded."
        
    # Convert BGR to RGB since torch.hub models usually expect RGB, although YOLOv5 handles BGR natively too.
    # YOLOv5's AutoShape handles cv2 images (which are BGR) correctly by default.
    try:
        results = model(image)
    except Exception as e:
        return image, None, f"❌ Model inference failed: {str(e)}"
        
    predictions = results.pandas().xyxy[0]
    
    if len(predictions) == 0:
        return image, None, "❌ No document content detected by DALAI model."
        
    # Draw detections on the image
    vis_img = image.copy()
    
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0.0, 0.0
    
    for idx, row in predictions.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        conf = row['confidence']
        
        # Update global bounding box
        min_x, min_y = min(min_x, xmin), min(min_y, ymin)
        max_x, max_y = max(max_x, xmax), max(max_y, ymax)
        
        # Draw bounding box and label
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (255, 100, 0), 2)
        cv2.putText(vis_img, f"{label} {conf:.2f}", (xmin, max(ymin - 10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
                    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    
    # Add a small padding (e.g. 5%) around the content bounding box to capture edges
    pad_x = int((max_x - min_x) * 0.05)
    pad_y = int((max_y - min_y) * 0.05)
    
    min_x = max(0, min_x - pad_x)
    min_y = max(0, min_y - pad_y)
    max_x = min(w, max_x + pad_x)
    max_y = min(h, max_y + pad_y)
    
    # Draw the derived document boundary in red
    cv2.rectangle(vis_img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 4)
    cv2.putText(vis_img, "Derived Document Crop", (min_x, max(min_y - 10, 0)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
    # Crop the image
    cropped_img = image[min_y:max_y, min_x:max_x]
    
    return vis_img, cropped_img, "✅ Document content detected and cropped successfully!"
