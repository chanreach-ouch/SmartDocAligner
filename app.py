import gradio as gr
import cv2
import numpy as np
import traceback
import os
import tempfile

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
    Detects document corners using classical OpenCV methods.
    Uses multiple strategies in order of preference:
      1. Canny edges at several threshold pairs + varying approxPolyDP epsilon
      2. Morphological dilation of edges before contouring
      3. Fallback: bounding-box corners of the single largest contour
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        min_area = (h * w) * 0.05  # ignore tiny contours

        def _find_quad(edged):
            """Try to find a 4-point contour in the edge image."""
            contours, _ = cv2.findContours(
                edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            for c in contours:
                if cv2.contourArea(c) < min_area:
                    continue
                peri = cv2.arcLength(c, True)
                # Try a range of epsilons from tight to loose
                for eps in [0.01, 0.02, 0.03, 0.05]:
                    approx = cv2.approxPolyDP(c, eps * peri, True)
                    if len(approx) == 4:
                        return approx.reshape(4, 2)
            return None

        def _bounding_quad(edged):
            """Fallback: use bounding rect of the largest meaningful contour."""
            contours, _ = cv2.findContours(
                edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            if not contours:
                return None
            c = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(c)
            return np.array([[x, y], [x+cw, y], [x+cw, y+ch], [x, y+ch]], dtype=np.float32)

        # Strategy 1 – multiple Canny thresholds
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        for lo, hi in [(50, 150), (75, 200), (30, 100), (100, 250)]:
            edged = cv2.Canny(blur, lo, hi)
            result = _find_quad(edged)
            if result is not None:
                return result

        # Strategy 2 – dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for lo, hi in [(50, 150), (30, 100)]:
            edged = cv2.Canny(blur, lo, hi)
            edged = cv2.dilate(edged, kernel, iterations=1)
            result = _find_quad(edged)
            if result is not None:
                return result

        # Strategy 3 – adaptive threshold then contours
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        result = _find_quad(thresh)
        if result is not None:
            return result

        # Strategy 4 – bounding box of the largest contour as a last resort
        edged = cv2.Canny(blur, 50, 150)
        return _bounding_quad(edged)

    except Exception as e:
        print(f"OpenCV detection error: {e}")
        return None


def detect_document_orientation(image):
    """
    Estimates document rotation (0, 90, 180, 270 degrees) using horizontal
    and vertical gradient energy projection analysis — no external model needed.
    Returns (rotated_image, angle_degrees, message).
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Compute gradient magnitude in both axes
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy_h = np.mean(np.abs(gx))   # horizontal line energy → vertical edges
        energy_v = np.mean(np.abs(gy))   # vertical line energy → horizontal edges

        h, w = gray.shape
        aspect = w / h

        # Use horizontal projection profile to detect text baseline direction
        # Blur and threshold to get text-like blobs
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Row-wise and column-wise sum profiles
        row_profile = np.sum(binary, axis=1).astype(float)
        col_profile = np.sum(binary, axis=0).astype(float)

        row_variance = np.var(row_profile)
        col_variance = np.var(col_profile)

        # High row variance → clear horizontal text lines → likely upright or 180°
        # High col variance → vertical text lines → likely 90° or 270°
        if row_variance >= col_variance:
            # Image appears to have horizontal text lines
            if aspect >= 0.7:  # wide enough → portrait/landscape upright
                angle = 0
                label = "upright (0°)"
            else:
                # Very tall image with horizontal lines — probably 90° or 270°
                angle = 90
                label = "rotated 90°"
        else:
            # Vertical text lines → rotated
            if aspect <= 1.0:
                angle = 90
                label = "rotated 90°"
            else:
                angle = 270
                label = "rotated 270°"

        # Apply correction rotation
        if angle == 0:
            corrected = image.copy()
        elif angle == 90:
            corrected = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            corrected = cv2.rotate(image, cv2.ROTATE_180)
        else:  # 270
            corrected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        msg = f"✅ Detected orientation: {label}. Correction applied."
        return corrected, angle, msg

    except Exception as e:
        return image, 0, f"❌ Orientation detection failed: {e}"


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
        corrected, angle, orient_msg = detect_document_orientation(image)
        res_msg.append(f"Document orientation: {orient_msg}")
        # Show original with angle label as visualization, corrected as crop
        vis1 = image.copy()
        label_text = f"Detected angle: {angle} deg"
        cv2.putText(vis1, label_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        crop1 = corrected
        
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
