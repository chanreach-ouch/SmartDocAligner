import os
import glob
import xml.etree.ElementTree as ET
import cv2
import json
import numpy as np
from shapely.geometry import Polygon
import traceback

def parse_ground_truth(xml_path):
    """Parses a .gt.xml file, returns dict mapping non-rejected frame index to 4 corner points."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frames = {}
    
    seg_results = root.find('segmentation_results')
    if seg_results is None:
        return frames
        
    for frame in seg_results.findall('frame'):
        if frame.get('rejected') == 'true':
            continue
        index = int(frame.get('index'))
        points = {}
        for point in frame.findall('point'):
            name = point.get('name')
            x = float(point.get('x'))
            y = float(point.get('y'))
            points[name] = (x, y)
        if len(points) == 4 and all(k in points for k in ['tl', 'tr', 'br', 'bl']):
            frames[index] = points
    return frames

def extract_frames(video_path, frame_indices):
    """Extracts specific frames from a video by index (1-based index)."""
    cap = cv2.VideoCapture(video_path)
    extracted = {}
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ret, frame = cap.read()
        if ret:
            extracted[idx] = frame
    cap.release()
    return extracted

def compute_iou(pred_pts, gt_pts):
    p1 = Polygon(pred_pts)
    p2 = Polygon(gt_pts)
    if not p1.is_valid: p1 = p1.convex_hull
    if not p2.is_valid: p2 = p2.convex_hull
    try:
        iou = p1.intersection(p2).area / p1.union(p2).area
    except:
        iou = 0.0
    return iou

def main():
    try:
        import turbojpeg
        class DummyTurboJPEG:
            def __init__(self, *args, **kwargs): pass
        turbojpeg.TurboJPEG = DummyTurboJPEG
    except ImportError:
        pass

    try:
        from docaligner import DocAligner
    except ImportError:
        print("Please install docaligner-docsaid")
        return

    # Initialize model
    print("Initializing DocAligner...")
    try:
        model = DocAligner(model_cfg='fastvit_sa24')
        print("Initialized DocAligner with fastvit_sa24 backbone")
    except Exception as e:
        print(f"Failed to load fastvit_sa24: {e}. Trying lcnet...")
        try:
            model = DocAligner(model_cfg='lcnet')
            print("Initialized DocAligner with lcnet backbone")
        except:
            model = DocAligner()
            print("Initialized DocAligner with default backbone")
            
    base_dir = r"C:\Users\Chanreach\Documents\SmartDocAligner\data"
    videos_dir = os.path.join(base_dir, "sampleDataset", "input_sample", "background00")
    gt_dir = os.path.join(base_dir, "sampleDataset", "input_sample_groundtruth", "background00_gt")
    out_dir = os.path.join(base_dir, "extracted")
    vis_dir = os.path.join(out_dir, "visualizations")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    videos = glob.glob(os.path.join(videos_dir, "*.avi"))
    
    all_results = []
    manifest = {}
    
    sample_interval = 10
    
    print(f"Found {len(videos)} videos. Processing...")
    
    for video_path in videos:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        gt_path = os.path.join(gt_dir, f"{stem}.gt.xml")
        
        if not os.path.exists(gt_path):
            print(f"Skipping {stem}, ground truth not found.")
            continue
            
        gt_data = parse_ground_truth(gt_path)
        if not gt_data:
            continue
            
        # Sample every Nth frame from the valid frames
        valid_indices = sorted(list(gt_data.keys()))
        sampled_indices = valid_indices[::sample_interval]
        
        print(f"Video {stem}: extracting {len(sampled_indices)} frames...")
        frames = extract_frames(video_path, sampled_indices)
        
        for idx, frame in frames.items():
            img_filename = f"{stem}_frame{idx}.jpg"
            img_path = os.path.join(out_dir, img_filename)
            cv2.imwrite(img_path, frame)
            
            # Format ground truth: tl, tr, br, bl
            pts = gt_data[idx]
            gt_array = np.array([pts['tl'], pts['tr'], pts['br'], pts['bl']], dtype=np.float32)
            manifest[img_filename] = gt_array.tolist()
            
            # Inference
            try:
                # The model's return value depends on the API version, usually it returns the polygon.
                preds = model(frame)
                
                # Check for dict vs direct array
                if isinstance(preds, dict):
                    # In some versions, output might be dict with 'points' or similar
                    if 'polygon' in preds:
                        pred_array = np.array(preds['polygon'])
                    else:
                        # Fallback for unknown dict structure
                        print(f"Warning: Unexpected dict keys from model: {preds.keys()}")
                        continue
                else:
                    pred_array = np.array(preds)
                
                if pred_array.shape != (4, 2):
                    print(f"Warning: predictions for {img_filename} not 4 points: {pred_array.shape}")
                    continue
                
                # Compute metrics
                pixel_error = np.mean(np.linalg.norm(pred_array - gt_array, axis=1))
                iou = compute_iou(pred_array, gt_array)
                
                all_results.append({
                    'filename': img_filename,
                    'img_path': img_path,
                    'frame': frame,
                    'gt': gt_array,
                    'pred': pred_array,
                    'error': float(pixel_error),
                    'iou': float(iou)
                })
            except Exception as e:
                print(f"Error processing {img_filename}: {e}")
                traceback.print_exc()
                
    if not all_results:
        print("No valid results computed.")
        return
        
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    # Sort by error (worst first)
    all_results.sort(key=lambda x: x['error'], reverse=True)
    
    print("\n" + "="*50)
    print("SUMMARY REPORT")
    print("="*50)
    print(f"Total images tested: {len(all_results)}")
    
    avg_error = np.mean([r['error'] for r in all_results])
    avg_iou = np.mean([r['iou'] for r in all_results])
    
    print(f"Average pixel error: {avg_error:.2f} pixels")
    print(f"Average IoU:         {avg_iou:.4f}")
    
    iou_arr = np.array([r['iou'] for r in all_results])
    good = np.sum(iou_arr > 0.9)
    okay = np.sum((iou_arr > 0.7) & (iou_arr <= 0.9))
    poor = np.sum(iou_arr <= 0.7)
    total = len(all_results)
    
    print(f"% IoU > 0.9 (good):      {good/total*100:.1f}%")
    print(f"% IoU 0.7-0.9 (okay):    {okay/total*100:.1f}%")
    print(f"% IoU < 0.7 (poor):      {poor/total*100:.1f}%")
    
    print("\nWorst 3 performing images (by pixel error):")
    for i in range(min(3, len(all_results))):
        r = all_results[i]
        print(f"  {r['filename']}: Error={r['error']:.2f}, IoU={r['iou']:.4f}")
        
    # Save visual comparisons for a mix of worst and best
    vis_samples = all_results[:5] + all_results[-5:]
    for i, r in enumerate(vis_samples):
        vis_img = r['frame'].copy()
        gt = r['gt'].astype(np.int32)
        pred = r['pred'].astype(np.int32)
        
        cv2.polylines(vis_img, [gt], True, (0, 255, 0), 2) # Green GT
        cv2.polylines(vis_img, [pred], True, (0, 0, 255), 2) # Red Pred
        
        cv2.imwrite(os.path.join(vis_dir, f"vis_{i}_{r['filename']}"), vis_img)
    
    print(f"\nVisualizations saved to {vis_dir}")

if __name__ == "__main__":
    main()
