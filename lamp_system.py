import asyncio
import cv2
import numpy as np
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from mediapipe.python.solutions.face_mesh import FaceMesh

# Import custom modules
import lamp_memory
import lamp_brain
from lamp_vision import LampVision

class LeLampSystem:
    def __init__(self):
        self.current_state = "DISENGAGED"
        self.event_queue = asyncio.Queue()
        
        # Calibration thresholds
        self.YAW_THRESHOLD = 30
        self.PITCH_MIN = -180
        self.PITCH_MAX = -150
        
        # Initialize modules
        lamp_memory.init_db()
        self.detector = LampVision()

    def _run_camera_pipeline(self, loop):
        """Background thread handling face tracking and live object detection."""
        cap = cv2.VideoCapture(0)
        
        face_mesh = FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        MODEL_POINTS = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]
        
        local_engagement = "DISENGAGED"
        look_away_start_time = None
        last_scan_time = 0
        
        print("[Perception Thread] Camera stream active. Face mesh and YOLO online.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            current_gaze_engaged = False
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    image_points = np.array([[face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] for idx in LANDMARK_INDICES], dtype=np.float64)
                    focal_length = w
                    camera_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype=np.float64)
                    success_pnp, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
                    
                    if success_pnp:
                        rmat, _ = cv2.Rodrigues(rvec)
                        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvec)))
                        pitch, yaw = euler_angles[0, 0], euler_angles[1, 0]
                        
                        if abs(yaw) < self.YAW_THRESHOLD and (self.PITCH_MIN <= pitch <= self.PITCH_MAX):
                            current_gaze_engaged = True
            
            if current_gaze_engaged:
                look_away_start_time = None
                if local_engagement == "DISENGAGED":
                    local_engagement = "ENGAGED"
                    asyncio.run_coroutine_threadsafe(self.event_queue.put("USER_ENGAGED"), loop)
                    
                # Scan environment every 3 seconds while engaged
                current_time = time.time()
                if current_time - last_scan_time > 3.0:
                    last_scan_time = current_time
                    detected_items = self.detector.scan_frame(frame)
                    
                    for item in detected_items:
                        asyncio.run_coroutine_threadsafe(
                            lamp_memory.log_detected_object(item["label"], item["confidence"], item["x"], item["y"], "ENGAGED"),
                            loop
                        )
            else:
                if local_engagement == "ENGAGED":
                    if look_away_start_time is None:
                        look_away_start_time = time.time()
                    elif time.time() - look_away_start_time > 1.5:
                        local_engagement = "DISENGAGED"
                        look_away_start_time = None
                        asyncio.run_coroutine_threadsafe(self.event_queue.put("USER_DISENGAGED"), loop)

            color = (0, 255, 0) if local_engagement == "ENGAGED" else (0, 0, 255)
            cv2.putText(frame, f"SYSTEM STATE: {local_engagement}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("LeLamp Live Vision & Perception System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                asyncio.run_coroutine_threadsafe(self.event_queue.put("SYSTEM_SHUTDOWN"), loop)
                break
                
        cap.release()
        cv2.destroyAllWindows()

    async def user_input_listener(self):
        """Asynchronously listens for keyboard input without blocking the camera thread."""
        loop = asyncio.get_running_loop()
        print("\n [Input Loop] Active. Type your question in the terminal when ENGAGED (e.g., 'Where is my phone?')")
        
        while self.current_state != "SHUTDOWN":
            # Run the blocking input() function inside a separate thread pool executor
            user_query = await loop.run_in_executor(None, input, "💬 Ask LeLamp: ")
            
            if user_query.strip().lower() in ['exit', 'quit', 'q']:
                await self.event_queue.put("SYSTEM_SHUTDOWN")
                break
                
            if self.current_state == "ENGAGED":
                # Send the query right to our Ollama + SQLite engine!
                response = await lamp_brain.think_and_respond(user_query)
                print(f"\n[LeLamp]: {response}\n")
            else:
                print("\n [LeLamp]: (System is DISENGAGED. Look at the camera first so I can open my attention loop!)\n")

    async def behavioral_fsm(self):
        """The core central decision maker of the lamp."""
        print("[Brain FSM] Decision loop online.")
        while True:
            event = await self.event_queue.get()
            
            if event == "USER_ENGAGED" and self.current_state != "ENGAGED":
                self.current_state = "ENGAGED"
                print("\n ** [LAMP STATUS] -> Looking directly at user. Interactive terminal listening... **\n")
                
            elif event == "USER_DISENGAGED" and self.current_state == "ENGAGED":
                self.current_state = "IDLE_ATTENTION"
                print("\n ** [LAMP STATUS] -> User looked away. Input locked until re-engaged. **\n")
                
            elif event == "SYSTEM_SHUTDOWN":
                self.current_state = "SHUTDOWN"
                print("[System] Shutting down execution engines...")
                break
                
            self.event_queue.task_done()

    async def main(self):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            camera_task = loop.run_in_executor(executor, self._run_camera_pipeline, loop)
            await asyncio.gather(
                self.behavioral_fsm(),
                self.user_input_listener(),
                camera_task
            )

if __name__ == "__main__":
    system = LeLampSystem()
    try:
        asyncio.run(system.main())
    except KeyboardInterrupt:
        print("\n[System] Graceful exit executed.")