import gradio as gr
import cv2
import numpy as np
import traceback
import os

# IMPORTANT: Patch TurboJPEG BEFORE importing docaligner/capybara.
# capybara instantiates TurboJPEG() at module-import time, which fails
# on Windows when the native library is missing.
try:
    import turbojpeg
    class DummyTurboJPEG:
        def __init__(self, *args, **kwargs): pass
        def decode(self, *a, **kw): raise NotImplementedError
        def encode(self, *a, **kw): raise NotImplementedError
    turbojpeg.TurboJPEG = DummyTurboJPEG
except ImportError:
    pass

from docaligner import DocAligner
from segmentation_model import detect_document_segmentation

# Absolute path to the models directory so ONNX loading works regardless of CWD
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

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

def detect_document_corners_opencv(image):
    """
    Detects document corners using classical OpenCV methods (Canny Edge + Contours).
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Apply Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Edge detection
        edged = cv2.Canny(blur, 75, 200)

        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Sort contours by area, keeping only the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # If our approximated contour has four points, we can assume we found the document
            if len(approx) == 4:
                return approx.reshape(4, 2)
                
        return None
    except Exception as e:
        print(f"OpenCV detection error: {e}")
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
        return "Please upload an image.", gr.update(visible=False), gr.update(), None, None, gr.update(visible=False), gr.update(), None, None
        
    # Ensure image is RGB (Gradio sometimes passes RGBA)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
    res_msg = []
    
    # Defaults
    vis1, crop1 = None, None
    vis2, crop2 = None, None
    title1, title2 = f"### {model_choice} Output", ""
    show_col2 = False
    
    if model_choice == "Compare both":
        # Run DocAligner
        corners = detect_document_corners(image)
        if corners is None:
            res_msg.append("DocAligner: ❌ Failed to detect document corners.")
            vis1 = image.copy()
        else:
            vis1 = image.copy()
            corners_int = corners.astype(np.int32)
            cv2.polylines(vis1, [corners_int], isClosed=True, color=(0, 255, 0), thickness=4)
            for pt in corners_int:
                cv2.circle(vis1, tuple(pt), radius=8, color=(255, 0, 0), thickness=-1)
            crop1 = crop_document(image, corners)
            res_msg.append("DocAligner: ✅ Document detected.")
            
        # Run DALAI
        vis2, crop2, d_msg = detect_document_segmentation(image)
        res_msg.append(f"DALAI: {d_msg}")
        
        title1 = "### DocAligner Output"
        title2 = "### DALAI Output"
        show_col2 = True

    elif model_choice == "DocAligner (corner detection)":
        corners = detect_document_corners(image)
        if corners is None:
            res_msg.append("DocAligner: ❌ Failed to detect document corners.")
            vis1 = image.copy()
        else:
            vis1 = image.copy()
            corners_int = corners.astype(np.int32)
            cv2.polylines(vis1, [corners_int], isClosed=True, color=(0, 255, 0), thickness=4)
            for pt in corners_int:
                cv2.circle(vis1, tuple(pt), radius=8, color=(255, 0, 0), thickness=-1)
            crop1 = crop_document(image, corners)
            res_msg.append("DocAligner: ✅ Document detected.")
            
    elif model_choice == "Document Segmentation (DALAI)":
        vis1, crop1, d_msg = detect_document_segmentation(image)
        res_msg.append(f"DALAI: {d_msg}")
        
    elif model_choice == "OpenCV crop method":
        corners = detect_document_corners_opencv(image)
        if corners is None:
            res_msg.append("OpenCV: ❌ Failed to detect document corners.")
            vis1 = image.copy()
        else:
            vis1 = image.copy()
            corners_int = corners.astype(np.int32)
            cv2.polylines(vis1, [corners_int], isClosed=True, color=(0, 255, 0), thickness=4)
            for pt in corners_int:
                cv2.circle(vis1, tuple(pt), radius=8, color=(255, 0, 0), thickness=-1)
            crop1 = crop_document(image, corners)
            res_msg.append("OpenCV: ✅ Document detected.")
            
    elif model_choice == "UVDoc inference model":
        import tempfile
        import os
        from document_unwarping.paddle_uvdoc import unwarp_with_paddle
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_out:
            in_path = f_in.name
            out_path = f_out.name
            
        cv2.imwrite(in_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        res = unwarp_with_paddle(in_path, out_path)
        
        if res.get("error"):
            res_msg.append(f"Paddle UVDoc Error: {res['error']}")
            vis1 = image.copy()
        else:
            res_msg.append(f"Paddle UVDoc: ✅ Unwarped ({res.get('inference_time_ms', 0):.1f}ms)")
            crop1 = cv2.cvtColor(cv2.imread(out_path), cv2.COLOR_BGR2RGB)
            vis1 = image.copy()
            
        try: os.remove(in_path); os.remove(out_path)
        except: pass

    elif model_choice == "UVDoc ONNX model":
        import tempfile
        from document_unwarping.onnx_uvdoc import unwarp_with_onnx
        
        onnx_model_path = os.path.join(_MODELS_DIR, "UVDoc_infer.onnx")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_out:
            in_path = f_in.name
            out_path = f_out.name
            
        cv2.imwrite(in_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        res = unwarp_with_onnx(in_path, out_path, model_path=onnx_model_path)
        
        if res.get("error"):
            res_msg.append(f"ONNX UVDoc Error: {res['error']}")
            vis1 = image.copy()
        else:
            res_msg.append(f"ONNX UVDoc: ✅ Unwarped ({res.get('inference_time_ms', 0):.1f}ms)")
            crop1 = cv2.cvtColor(cv2.imread(out_path), cv2.COLOR_BGR2RGB)
            vis1 = image.copy()
            
        try: os.remove(in_path); os.remove(out_path)
        except: pass
        
    elif model_choice == "Document orientation model":
        res_msg.append(f"{model_choice}: ⚠️ Placeholder. Model weights and inference code not yet provided.")
        vis1 = image.copy()
        
    return (
        "\n".join(res_msg), 
        gr.update(visible=True), gr.update(value=title1), vis1, crop1,
        gr.update(visible=show_col2), gr.update(value=title2), vis2, crop2
    )

# Build Gradio UI 
with gr.Blocks(title="Document Detection Tester") as demo:
    gr.Markdown("# 📄 Document Detection Tester")
    gr.Markdown("Compare **DocAligner** (corner detection) against **DALAI** (content segmentation) for document extraction.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Image", type="numpy")
            model_choice = gr.Radio(
                choices=[
                    "DocAligner (corner detection)", 
                    "Document Segmentation (DALAI)", 
                    "Compare both",
                    "UVDoc inference model",
                    "UVDoc ONNX model",
                    "Document orientation model",
                    "OpenCV crop method"
                ],
                value="Compare both",
                label="Select Detection Model"
            )
            submit_btn = gr.Button("Detect & Crop", variant="primary")
            output_msg = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column(visible=False) as col_1:
                    title_1 = gr.Markdown("### Result")
                    output_vis_1 = gr.Image(label="Visualization")
                    output_crop_1 = gr.Image(label="Crop")
                with gr.Column(visible=False) as col_2:
                    title_2 = gr.Markdown("### Result 2")
                    output_vis_2 = gr.Image(label="Visualization 2")
                    output_crop_2 = gr.Image(label="Crop 2")
            
    submit_btn.click(
        fn=process_image,
        inputs=[input_img, model_choice],
        outputs=[
            output_msg, 
            col_1, title_1, output_vis_1, output_crop_1,
            col_2, title_2, output_vis_2, output_crop_2
        ]
    )

if __name__ == "__main__":
    demo.launch()
