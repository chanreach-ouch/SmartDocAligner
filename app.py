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
from segmentation_model import detect_document_segmentation

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

def process_image(image, model_choice):
    """
    Gradio interface function.
    """
    if image is None:
        return "Please upload an image.", None, None, None, None
        
    res_msg = []
    
    doc_vis, doc_crop = None, None
    dalai_vis, dalai_crop = None, None
    
    if model_choice in ["DocAligner (corner detection)", "Compare both"]:
        corners = detect_document_corners(image)
        if corners is None:
            res_msg.append("DocAligner: ❌ Failed to detect document corners.")
            doc_vis = image.copy()
        else:
            doc_vis = image.copy()
            corners_int = corners.astype(np.int32)
            cv2.polylines(doc_vis, [corners_int], isClosed=True, color=(0, 255, 0), thickness=4)
            for pt in corners_int:
                cv2.circle(doc_vis, tuple(pt), radius=8, color=(255, 0, 0), thickness=-1)
            doc_crop = crop_document(image, corners)
            res_msg.append("DocAligner: ✅ Document detected.")
            
    if model_choice in ["Document Segmentation (DALAI)", "Compare both"]:
        d_vis, d_crop, d_msg = detect_document_segmentation(image)
        dalai_vis = d_vis
        dalai_crop = d_crop
        res_msg.append(f"DALAI: {d_msg}")
        
    return "\n".join(res_msg), doc_vis, doc_crop, dalai_vis, dalai_crop

# Build Gradio UI 
with gr.Blocks(title="Document Detection Tester") as demo:
    gr.Markdown("# 📄 Document Detection Tester")
    gr.Markdown("Compare **DocAligner** (corner detection) against **DALAI** (content segmentation) for document extraction.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Image", type="numpy")
            model_choice = gr.Radio(
                choices=["DocAligner (corner detection)", "Document Segmentation (DALAI)", "Compare both"],
                value="Compare both",
                label="Select Detection Model"
            )
            submit_btn = gr.Button("Detect & Crop", variant="primary")
            output_msg = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### DocAligner Output")
                    output_vis_doc = gr.Image(label="DocAligner Corners")
                    output_crop_doc = gr.Image(label="DocAligner Crop")
                with gr.Column():
                    gr.Markdown("### DALAI Output")
                    output_vis_dalai = gr.Image(label="DALAI Segmentation")
                    output_crop_dalai = gr.Image(label="DALAI Crop")
            
    submit_btn.click(
        fn=process_image,
        inputs=[input_img, model_choice],
        outputs=[output_msg, output_vis_doc, output_crop_doc, output_vis_dalai, output_crop_dalai]
    )

if __name__ == "__main__":
    demo.launch()
