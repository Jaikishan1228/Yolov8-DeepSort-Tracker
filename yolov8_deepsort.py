"""
YOLOv8 + DeepSORT Aircraft Tracker - Simplified
Only shows IDs when targets detected, reuses old IDs
"""
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path
import csv
from datetime import datetime


class AircraftTracker:
    """Simplified aircraft tracker with persistent IDs"""
    
    def __init__(self, model_path=None, results_dir=None, conf_threshold=0.25):
        # Paths
        self.base_dir = Path(__file__).parent
        self.model_path = model_path or self.base_dir / "aircraft2" / "weights" / "best.pt"
        self.results_dir = results_dir or self.base_dir / "Results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        
        # Load model
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        
        # Aircraft classes
        self.class_names = [
            'A10', 'A400M', 'AG600', 'AV8B', 'B1', 'B2', 'B52', 'Be200',
            'C130', 'C17', 'C2', 'C5', 'E2', 'E7', 'EF2000', 'F117',
            'F14', 'F15', 'F16', 'F18', 'F22', 'F35', 'F4', 'J10',
            'J20', 'JAS39', 'KC135', 'MQ9', 'Mig31', 'Mirage2000', 'P3', 'RQ4',
            'Rafale', 'SR71', 'Su24', 'Su34', 'Su57', 'Tornado', 'Tu160', 'Tu95',
            'U2', 'US2', 'V22', 'Vulcan', 'XB70', 'YF23'
        ]
        
        # DeepSORT - Only show when detected
        self.tracker = DeepSort(
            max_age=60,               # Keep tracks for 2 seconds
            n_init=1,                 # Immediate confirmation
            max_iou_distance=0.4,     # Flexible spatial matching
            max_cosine_distance=0.3,  # Flexible appearance matching
            nn_budget=300,            # Large feature memory
            embedder="mobilenet",
            embedder_gpu=torch.cuda.is_available()
        )
        
        self.track_history = {}
    
    def reset_tracker(self):
        """Reset tracker for new video"""
        self.tracker = DeepSort(
            max_age=60, n_init=1, max_iou_distance=0.4,
            max_cosine_distance=0.3, nn_budget=300,
            embedder="mobilenet", embedder_gpu=torch.cuda.is_available()
        )
        self.track_history = {}
    
    def process_video(self, video_path, output_path=None, display=True):
        """Process video with persistent ID tracking"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.results_dir / f"{Path(video_path).stem}_tracked_{timestamp}.mp4"
        
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        print(f"\nProcessing: {Path(video_path).name}")
        self.reset_tracker()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLO detection with balanced NMS
            results = self.model(frame, conf=self.conf_threshold, iou=0.5, verbose=False, device=self.device)
            detections = results[0].boxes
            
            # Prepare for DeepSORT
            deepsort_dets = []
            if len(detections) > 0:
                for box in detections:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                    deepsort_dets.append((bbox, conf, cls))
            
            # Update tracker - reuses old IDs automatically
            tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
            
            # Draw ONLY active detections (not predicted tracks)
            active_tracks = 0
            for track in tracks:
                # Only show if confirmed AND has recent detection (time_since_update == 0)
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                # Get class
                class_name = "Aircraft"
                if hasattr(track, 'det_class') and track.det_class is not None:
                    class_idx = int(track.det_class)
                    if class_idx < len(self.class_names):
                        class_name = self.class_names[class_idx]
                
                # Update history
                if track_id not in self.track_history:
                    self.track_history[track_id] = {
                        'first': frame_count, 'last': frame_count,
                        'count': 1, 'class': class_name
                    }
                else:
                    self.track_history[track_id]['last'] = frame_count
                    self.track_history[track_id]['count'] += 1
                
                # Draw box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id} {class_name}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                active_tracks += 1
            
            # Frame info
            cv2.putText(frame, f"Frame:{frame_count} | Tracks:{active_tracks}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writer.write(frame)
            
            if display:
                # Smaller window for easier viewing
                cv2.imshow("Tracker", cv2.resize(frame, None, fx=0.6, fy=0.6))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Save CSV
        self._save_csv(Path(video_path).stem)
        
        print(f"Done! Frames: {frame_count} | Unique Tracks: {len(self.track_history)}")
        print(f"Output: {output_path}\n")
        
        return {'frames_processed': frame_count, 'total_tracks': len(self.track_history), 'output_path': output_path}
    
    def process_webcam(self, camera_id=0, display=True):
        """Process webcam with persistent tracking"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return None
        
        print(f"\nWebcam tracking started. Press 'q' to quit.")
        self.reset_tracker()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLO detection with balanced NMS
            results = self.model(frame, conf=self.conf_threshold, iou=0.5, verbose=False, device=self.device)
            detections = results[0].boxes
            
            # Prepare for DeepSORT
            deepsort_dets = []
            if len(detections) > 0:
                for box in detections:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                    deepsort_dets.append((bbox, conf, cls))
            
            # Update tracker
            tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
            
            # Draw only active detections (not predicted)
            active_tracks = 0
            for track in tracks:
                # Only show if confirmed AND has recent detection
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                # Get class
                class_name = "Aircraft"
                if hasattr(track, 'det_class') and track.det_class is not None:
                    class_idx = int(track.det_class)
                    if class_idx < len(self.class_names):
                        class_name = self.class_names[class_idx]
                
                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id} {class_name}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                active_tracks += 1
            
            cv2.putText(frame, f"Tracks: {active_tracks}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if display:
                cv2.imshow("Webcam Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"\nStopped. Frames: {frame_count}")
        return {'frames_processed': frame_count, 'total_tracks': len(self.track_history)}
    
    def _save_csv(self, video_name):
        """Save tracking results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.results_dir / f"{video_name}_tracking_{timestamp}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Track_ID', 'Aircraft_Type', 'First_Frame', 'Last_Frame', 'Total_Frames'])
            
            for track_id, data in sorted(self.track_history.items()):
                writer.writerow([
                    track_id, data['class'], data['first'],
                    data['last'], data['count']
                ])
        
        print(f"CSV saved: {csv_path}")


def get_tracker(model_path=None, results_dir=None, conf_threshold=0.25):
    """Create tracker instance"""
    return AircraftTracker(model_path, results_dir, conf_threshold)


if __name__ == "__main__":
    print("=" * 70)
    print("This is a library module. Use run_tracker.py to run tracking.")
    print("=" * 70)
