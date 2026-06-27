import cv2
from ultralytics import YOLO

class LampVision:
    def __init__(self):
        # Load the ultra-lightweight YOLOv8 Nano model (automatically downloads on first run)
        self.model = YOLO("yolov8n.pt")
        
        # COCO dataset indices for things typically found on a desk
        self.target_classes = {
            39: "bottle",
            41: "cup",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "phone",
            73: "book"
        }

    def scan_frame(self, frame):
        """Scans a single frame and returns a list of detected desk objects with coordinates."""
        results = self.model(frame, verbose=False)[0]
        detected_objects = []
        
        h, w, _ = frame.shape
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Only track objects we care about with > 40% confidence
            if class_id in self.target_classes and confidence > 0.40:
                # Get normalized center coordinates (0.0 to 1.0)
                xywh = box.xywhn[0].tolist()
                x_center, y_center = xywh[0], xywh[1]
                
                detected_objects.append({
                    "label": self.target_classes[class_id],
                    "confidence": confidence,
                    "x": round(x_center, 2),
                    "y": round(y_center, 2)
                })
                
        return detected_objects

# Quick standalone test to make sure it runs and downloads the model weights
if __name__ == "__main__":
    print("Initializing YOLO Engine...")
    detector = LampVision()
    print("YOLO Engine Ready! Ready to integrate into the main thread loop.")