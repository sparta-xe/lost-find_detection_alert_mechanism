from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        # Configure model for better small object detection
        self.model.overrides['conf'] = 0.15  # Lower confidence for small objects
        self.model.overrides['iou'] = 0.3    # Lower IoU for overlapping objects
        self.model.overrides['max_det'] = 500  # More detections per image
        self.model.overrides['imgsz'] = 640   # Higher resolution for small objects

    def detect(self, frame, camera_id, timestamp):
        # Preprocess frame for better small object detection
        processed_frame = self._preprocess_frame(frame)
        
        # Run detection with multiple scales for small objects
        detections = []
        
        # Original scale detection
        results = self.model(processed_frame, verbose=False)[0]
        detections.extend(self._extract_detections(results, camera_id, timestamp, scale=1.0))
        
        # Upscaled detection for very small objects
        if min(frame.shape[:2]) < 800:  # Only upscale if image is small
            upscaled_frame = cv2.resize(processed_frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            upscaled_results = self.model(upscaled_frame, verbose=False)[0]
            upscaled_detections = self._extract_detections(upscaled_results, camera_id, timestamp, scale=1.5)
            detections.extend(upscaled_detections)
        
        # Remove duplicate detections
        detections = self._remove_duplicates(detections)
        
        return detections
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for better detection of small objects."""
        # Enhance contrast and brightness for small objects
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Slight sharpening for small objects
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened (70% original, 30% sharpened)
        result = cv2.addWeighted(frame, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def _extract_detections(self, results, camera_id, timestamp, scale=1.0):
        """Extract detections from YOLO results."""
        detections = []
        
        if results.boxes is None:
            return detections
            
        for idx, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Adjust coordinates if upscaled
            if scale != 1.0:
                x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
            
            # Calculate object size for filtering
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Accept smaller objects with lower confidence
            min_conf = 0.15 if area < 1000 else 0.25  # Lower threshold for small objects
            
            if conf >= min_conf and width > 10 and height > 10:  # Minimum size filter
                detections.append({
                    "label": results.names[cls_id],
                    "class_name": results.names[cls_id],
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "detection_id": f"yolo_{timestamp:.3f}_{idx}_{scale}",
                    "area": area,
                    "scale": scale
                })

        return detections
    
    def _remove_duplicates(self, detections):
        """Remove duplicate detections using Non-Maximum Suppression."""
        if len(detections) <= 1:
            return detections
        
        # Group by class
        class_groups = {}
        for det in detections:
            cls_id = det['class_id']
            if cls_id not in class_groups:
                class_groups[cls_id] = []
            class_groups[cls_id].append(det)
        
        final_detections = []
        
        for cls_id, group in class_groups.items():
            if len(group) == 1:
                final_detections.extend(group)
                continue
            
            # Apply NMS within each class
            boxes = np.array([det['bbox'] for det in group])
            scores = np.array([det['confidence'] for det in group])
            
            # Convert to (x, y, w, h) format for NMS
            boxes_nms = np.column_stack([
                boxes[:, 0],  # x1
                boxes[:, 1],  # y1
                boxes[:, 2] - boxes[:, 0],  # width
                boxes[:, 3] - boxes[:, 1]   # height
            ])
            
            # Apply OpenCV NMS
            indices = cv2.dnn.NMSBoxes(
                boxes_nms.tolist(), 
                scores.tolist(), 
                score_threshold=0.15, 
                nms_threshold=0.4
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                for idx in indices:
                    final_detections.append(group[idx])
        
        return final_detections
