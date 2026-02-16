import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pyautogui
import threading
from queue import Queue
import math
import dotenv # Remove this import and the code for it if not using external camera

# Load camera URL from .env file, default to webcam if not provided
dotenv.load_dotenv()
camera = dotenv.get_key(dotenv.find_dotenv(), "CAMERA_URL")
if not camera:
    camera = 0  # Default to webcam if no URL is provided

# Disable pyautogui failsafe and set pause to 0 for faster response
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

debug_mode = True  # Set to True to enable debug visuals, False for normal operation
mouse_control_enabled = True # Set to True to enable mouse control, False to disable it
actions_enabled = True # Set to True to enable click and drag actions, False to only allow movement (requires mouse_control_enabled = True)

latest_result = None
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

def get_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

def finger_open(base_point, joint_point, tip_point, threshold=0.1):
    base_dist = get_distance(base_point, joint_point)
    tip_dist = get_distance(base_point, tip_point)
    return tip_dist > base_dist * (1 + threshold)
    
def check_finger_state(hand_landmarks):
    # You can find the landmark indices in the MediaPipe documentation (0-20, 0 is wrist)
    fingers = { 
        "Thumb": finger_open(hand_landmarks[17], hand_landmarks[2], hand_landmarks[4], threshold=-0.1),
        "Index": finger_open(hand_landmarks[0], hand_landmarks[6], hand_landmarks[8]),
        "Middle": finger_open(hand_landmarks[0],hand_landmarks[10], hand_landmarks[12]),
        "Ring": finger_open(hand_landmarks[0], hand_landmarks[14], hand_landmarks[16]),
        "Pinky": finger_open(hand_landmarks[0], hand_landmarks[18], hand_landmarks[20])
    }
    return fingers # Returns a dictionary with the state of each finger (True for open, False for closed)

# Queue for non-blocking mouse operations
mouse_action_queue = Queue()

# Track previous states to avoid repeated actions
previous_actions = {
    "left_click": False,
    "right_click": False,
    "middle_click": False,
    "dragging": False
}

# Worker thread to handle mouse operations without blocking the main loop
def mouse_worker():
    while True:
        try:
            action = mouse_action_queue.get(timeout=0.1)
            if action is None:  # Signal to stop
                break
            
            if action == "left_click":
                pyautogui.click()
            elif action == "right_click":
                pyautogui.rightClick()
            elif action == "middle_click":
                pyautogui.middleClick()
            elif action == "mouse_down":
                pyautogui.mouseDown(button='left')
            elif action == "mouse_up":
                pyautogui.mouseUp(button='left')
        except:
            continue

# Start the mouse worker thread as daemon
mouse_thread = threading.Thread(target=mouse_worker, daemon=True)
mouse_thread.start()

# Sensitivity settings for mouse movement
high_sens = 1.5
low_sens = 0.5

# Initialize MediaPipe
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback
)

landmarker = HandLandmarker.create_from_options(options)

# Initialize video capture and set properties
cap = cv2.VideoCapture(camera)
cap.set(cv2.CAP_PROP_FPS, 30)

cv2.namedWindow('MediaPipe Mouse Control', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Mouse Control', 854, 480)

screen_width, screen_height = pyautogui.size()

prev_finger_x, prev_finger_y = None, None

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    if latest_result and latest_result.hand_landmarks:
        # Draw hand landmarks and connections for debugging
        if debug_mode:
            h, w, c = frame.shape
            hand_landmarks_list = latest_result.hand_landmarks
            
            # Connections between landmarks (0-20, 0 is wrist)
            hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),           # Index
                (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
                (0, 17), (17, 18), (18, 19), (19, 20)     # Pinky
            ]
            
            for hand_landmarks in hand_landmarks_list:
                points = []
                for idx, landmark in enumerate(hand_landmarks):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))
                    
                    # Draw landmark points
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                    
                    # Write landmark index
                    cv2.putText(frame, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Draw Connections
                for connection in hand_connections:
                    start_point = points[connection[0]]
                    end_point = points[connection[1]]
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                
                # Check finger states for debugging
                debug_finger_state = check_finger_state(hand_landmarks)
                y_pos = 50
                for finger_name, is_open in debug_finger_state.items():
                    status = "OPEN" if is_open else "CLOSED"
                    color = (0, 255, 0) if is_open else (0, 0, 255)
                    cv2.putText(frame, f"{finger_name}: {status}", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    y_pos += 35
        
        # Get finger state once per frame
        finger_state = check_finger_state(latest_result.hand_landmarks[0])
        
        if mouse_control_enabled and (finger_state["Index"] or finger_state["Pinky"]):
            # Determine which finger to use for movement and set sensitivity
            if finger_state["Index"]: # Index is for general movement
                finger_tip = latest_result.hand_landmarks[0][8]
                mouse_sensitivity = high_sens

            elif finger_state["Pinky"] and not finger_state["Index"]: # Pinky is for precise movement
                finger_tip = latest_result.hand_landmarks[0][20]
                mouse_sensitivity = low_sens
            
            # Take current finger coordinates
            curr_finger_x, curr_finger_y = finger_tip.x, finger_tip.y

            # Initialize previous finger position if not set
            if prev_finger_x is None or prev_finger_y is None:
                prev_finger_x, prev_finger_y = curr_finger_x, curr_finger_y
                continue 

            # Find the difference between current and previous finger positions
            dx = (curr_finger_x - prev_finger_x) * screen_width * mouse_sensitivity
            dy = (curr_finger_y - prev_finger_y) * screen_height * mouse_sensitivity

            # Get current mouse coordinates
            curr_mouse_x, curr_mouse_y = pyautogui.position()
            
            # Move mouse if the movement is significant enough to avoid jitter
            if abs(dx) > 3 or abs(dy) > 3:
                pyautogui.moveTo(curr_mouse_x + dx, curr_mouse_y + dy, duration=0)

            # Update previous finger position for the next iteration
            prev_finger_x, prev_finger_y = curr_finger_x, curr_finger_y

            # --- MOUSE ACTIONS ---
            if actions_enabled:
                LEFT_CLICK = finger_state["Thumb"] and not(finger_state["Middle"] or finger_state["Ring"] or finger_state["Pinky"])
                RIGHT_CLICK = finger_state["Middle"] and not(finger_state["Thumb"] or finger_state["Ring"] or finger_state["Pinky"])
                MIDDLE_CLICK = finger_state["Middle"] and finger_state["Ring"] and not(finger_state["Thumb"] or finger_state["Pinky"])
                DRAG = finger_state["Thumb"] and finger_state["Middle"] and finger_state["Ring"] and finger_state["Pinky"]
                DROP = not DRAG and previous_actions["dragging"]
                
                # Left Click
                if LEFT_CLICK and not previous_actions["left_click"]:
                    mouse_action_queue.put("left_click")
                    previous_actions["left_click"] = True
                elif not LEFT_CLICK and previous_actions["left_click"]:
                    previous_actions["left_click"] = False

                # Right Click
                if RIGHT_CLICK and not previous_actions["right_click"]:
                    mouse_action_queue.put("right_click")
                    previous_actions["right_click"] = True
                elif not RIGHT_CLICK and previous_actions["right_click"]:
                    previous_actions["right_click"] = False

                # Middle Click
                if MIDDLE_CLICK and not previous_actions["middle_click"]:
                    mouse_action_queue.put("middle_click")
                    previous_actions["middle_click"] = True
                elif not MIDDLE_CLICK and previous_actions["middle_click"]:
                    previous_actions["middle_click"] = False
                
                # Drag and Drop
                if DRAG and not previous_actions["dragging"]:
                    mouse_action_queue.put("mouse_down")
                    previous_actions["dragging"] = True
                elif DROP:
                    mouse_action_queue.put("mouse_up")
                    previous_actions["dragging"] = False
        
        else:
            # Reset initial position when no fingers are detected
            prev_finger_x, prev_finger_y = None, None 
    
    if debug_mode:
        cv2.imshow('MediaPipe Mouse Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # In non-debug mode, we can skip showing the window to save resources
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
mouse_action_queue.put(None)  # Signal worker thread to stop
mouse_thread.join(timeout=1)
landmarker.close()
cap.release()

cv2.destroyAllWindows()
