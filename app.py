import gradio as gr
import cv2
import numpy as np
import traceback

# Patch for TurboJPEG on Windows
try:
    import turbojpeg
    class DummyTurboJPEG:
        def __init__(self, *args, **kwargs): pass
    turbojpeg.TurboJPEG = DummyTurboJPEG
except ImportError:
    pass

from docaligner import DocAligner

# Initialize model once globally
try:
    print("Initializing DocAligner...")
    model = DocAligner(model_cfg='fastvit_sa24')
    print("DocAligner (fastvit_sa24) loaded successfully.")
except Exception as e:
    print(f"Failed to load fastvit_sa24, using default: {e}")
    try:
        model = DocAligner(model_cfg='lcnet')
    except:
        model = DocAligner()

def detect_document_corners(image):
    """
    Detects document corners in the provided image.
    Returns: numpy array of shape (4, 2) containing corner coordinates [tl, tr, br, bl]
    or None if detection fails.
    """
    try:
        preds = model(image)
        if isinstance(preds, dict) and 'polygon' in preds:
            pred_array = np.array(preds['polygon'])
        else:
            pred_array = np.array(preds)
            
        if pred_array.shape == (4, 2):
            return pred_array
        return None
    except Exception as e:
        print(f"Detection error: {e}")
        return None

def crop_document(image, corners):
    """
    Crops and applies perspective transform to extract the document.
    """
    # Order: tl, tr, br, bl
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sort points based on x and y to ensure consistent tl, tr, br, bl order if not already
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    
    # Compute width
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def process_image(image):
    """
    Gradio interface function.
    """
    if image is None:
        return None, None, "Please upload an image."
        
    corners = detect_document_corners(image)
    
    if corners is None:
        return image, None, "❌ Failed to detect a document in the image (no valid 4-point quad found). Try another photo."
        
    # Draw corners on original image
    vis_img = image.copy()
    corners_int = corners.astype(np.int32)
    cv2.polylines(vis_img, [corners_int], isClosed=True, color=(0, 255, 0), thickness=4)
    for pt in corners_int:
        cv2.circle(vis_img, tuple(pt), radius=8, color=(255, 0, 0), thickness=-1)
        
    # Crop the document
    cropped_img = crop_document(image, corners)
    
    return vis_img, cropped_img, "✅ Document detected successfully!"

# Build Gradio UI
with gr.Blocks(title="DocAligner Tester") as demo:
    gr.Markdown("# 📄 DocAligner Interactive Tester")
    gr.Markdown("Upload a photo of a paper document to test DocAligner's corner detection and auto-crop capabilities. The model uses the pretrained lightweight `fastvit_sa24` backbone.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Image", type="numpy")
            submit_btn = gr.Button("Detect & Crop", variant="primary")
        
        with gr.Column():
            output_msg = gr.Textbox(label="Status / Confidence", interactive=False)
            output_vis = gr.Image(label="Detected Corners")
            output_crop = gr.Image(label="Perspective Cropped Result")
            
    submit_btn.click(
        fn=process_image,
        inputs=[input_img],
        outputs=[output_vis, output_crop, output_msg]
    )

if __name__ == "__main__":
    demo.launch()
