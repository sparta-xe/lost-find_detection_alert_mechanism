from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        # Industry-standard configuration for small objects
        self.model.overrides['conf'] = 0.2    # Lowered for small objects
        self.model.overrides['iou'] = 0.5     # Higher IoU for better separation
        self.model.overrides['max_det'] = 1000  # More detections
        self.model.overrides['imgsz'] = 1280   # Higher resolution (industry standard)

    def detect(self, frame, camera_id, timestamp):
        # Layer 1: Enhanced preprocessing for small objects
        enhanced_frame = self._enhance_for_small_objects(frame)
        
        detections = []
        
        # Multi-scale detection (industry approach)
        scales = [1.0, 1.5, 2.0]  # Original, 1.5x, 2x upscaling
        
        for scale in scales:
            if scale != 1.0:
                h, w = enhanced_frame.shape[:2]
                scaled_frame = cv2.resize(enhanced_frame, (int(w * scale), int(h * scale)), 
                                        interpolation=cv2.INTER_CUBIC)
            else:
                scaled_frame = enhanced_frame
            
            # Industry-standard YOLO prediction
            results = self.model.predict(
                scaled_frame,
                conf=0.2,        # Low confidence for small objects
                imgsz=1280,      # High resolution
                iou=0.5,         # Better separation
                verbose=False
            )[0]
            
            scale_detections = self._extract_detections(results, camera_id, timestamp, scale)
            detections.extend(scale_detections)
        
        # Remove duplicates using NMS
        detections = self._advanced_nms(detections)
        
        return detections
    
    def _enhance_for_small_objects(self, frame):
        """Industry-standard enhancement for small object detection."""
        # 1. Contrast enhancement using CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 2. Sharpening filter for small objects
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 3. Blend original and sharpened (industry technique)
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def _extract_detections(self, results, camera_id, timestamp, scale=1.0):
        """Extract detections with scale adjustment."""
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                # Adjust coordinates for scale
                x1, y1, x2, y2 = box / scale
                
                # Get class name
                class_name = self.model.names[class_id]
                
                detection = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "label": class_name,
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "detection_id": f"{camera_id}_{timestamp}_{i}",
                    "scale_factor": scale
                }
                
                detections.append(detection)
        
        return detections
    
    def _advanced_nms(self, detections):
        """Advanced Non-Maximum Suppression for multi-scale detections."""
        if not detections:
            return detections
        
        # Group by class
        class_groups = {}
        for det in detections:
            label = det['label']
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(det)
        
        final_detections = []
        
        for label, group in class_groups.items():
            if len(group) <= 1:
                final_detections.extend(group)
                continue
            
            # Convert to format for cv2.dnn.NMSBoxes
            boxes = []
            confidences = []
            
            for det in group:
                x1, y1, x2, y2 = det['bbox']
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(det['confidence'])
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.5)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    final_detections.append(group[i])
        
        return final_detections
        
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
