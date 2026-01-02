import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import sys

# Windows-specific imports for system control
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    import screen_brightness_control as sbc
    WINDOWS_CONTROL_AVAILABLE = True
except ImportError:
    WINDOWS_CONTROL_AVAILABLE = False
    print("WARNING: Install pycaw and screen-brightness-control for full functionality")
    print("Run: pip install pycaw screen-brightness-control comtypes")

MODEL_PATH = "../models/gesture_random_forest_sklearn.pkl"

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Gesture to action mapping
GESTURE_ACTIONS = {
    'thumbs_up': 'volume_up',
    'thumbs_down': 'volume_down',
    'open_palm': 'brightness_up',
    'fist': 'brightness_down',
    'peace_sign': 'close_app'
}

# Control settings
VOLUME_STEP = 5          # Volume change per gesture (percentage)
BRIGHTNESS_STEP = 10     # Brightness change per gesture (percentage)
COOLDOWN_TIME = 0.1      # Seconds between actions to prevent rapid triggering
GESTURE_HOLD_FRAMES = 5  # Number of consecutive frames to confirm gesture


class WindowsController:
    """Handles Windows system controls for volume and brightness."""
    
    def __init__(self):
        self.volume_interface = None
        self.last_action_time = 0
        self.setup_audio()
    
    def setup_audio(self):
        """Initialize audio control interface."""
        if not WINDOWS_CONTROL_AVAILABLE:
            return
        
        try:
            devices = AudioUtilities.GetSpeakers()
            # Access the underlying device object
            interface = devices._dev.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
            print("Audio control initialized successfully")
        except Exception as e:
            print(f"Failed to initialize audio control: {e}")
            self.volume_interface = None
    
    def can_perform_action(self):
        """Check if enough time has passed since last action."""
        current_time = time.time()
        if current_time - self.last_action_time >= COOLDOWN_TIME:
            self.last_action_time = current_time
            return True
        return False
    
    def get_current_volume(self):
        """Get current system volume (0-100)."""
        if self.volume_interface is None:
            return 50
        try:
            return int(self.volume_interface.GetMasterVolumeLevelScalar() * 100)
        except:
            return 50
    
    def set_volume(self, level):
        """Set system volume (0-100)."""
        if self.volume_interface is None:
            print(f"[SIMULATED] Volume set to {level}%")
            return
        
        try:
            level = max(0, min(100, level))
            self.volume_interface.SetMasterVolumeLevelScalar(level / 100, None)
        except Exception as e:
            print(f"Failed to set volume: {e}")
    
    def volume_up(self):
        """Increase system volume."""
        if not self.can_perform_action():
            return None
        
        current = self.get_current_volume()
        new_level = min(100, current + VOLUME_STEP)
        self.set_volume(new_level)
        print(f"Volume UP: {current}% -> {new_level}%")
        return f"Volume: {new_level}%"
    
    def volume_down(self):
        """Decrease system volume."""
        if not self.can_perform_action():
            return None
        
        current = self.get_current_volume()
        new_level = max(0, current - VOLUME_STEP)
        self.set_volume(new_level)
        print(f"Volume DOWN: {current}% -> {new_level}%")
        return f"Volume: {new_level}%"
    
    def get_current_brightness(self):
        """Get current screen brightness (0-100)."""
        if not WINDOWS_CONTROL_AVAILABLE:
            return 50
        try:
            brightness = sbc.get_brightness()
            if isinstance(brightness, list):
                return brightness[0]
            return brightness
        except:
            return 50
    
    def set_brightness(self, level):
        """Set screen brightness (0-100)."""
        if not WINDOWS_CONTROL_AVAILABLE:
            print(f"[SIMULATED] Brightness set to {level}%")
            return
        
        try:
            level = max(0, min(100, level))
            sbc.set_brightness(level)
        except Exception as e:
            print(f"Failed to set brightness: {e}")
    
    def brightness_up(self):
        """Increase screen brightness."""
        if not self.can_perform_action():
            return None
        
        current = self.get_current_brightness()
        new_level = min(100, current + BRIGHTNESS_STEP)
        self.set_brightness(new_level)
        print(f"Brightness UP: {current}% -> {new_level}%")
        return f"Brightness: {new_level}%"
    
    def brightness_down(self):
        """Decrease screen brightness."""
        if not self.can_perform_action():
            return None
        
        current = self.get_current_brightness()
        new_level = max(0, current - BRIGHTNESS_STEP)
        self.set_brightness(new_level)
        print(f"Brightness DOWN: {current}% -> {new_level}%")
        return f"Brightness: {new_level}%"
    
    def execute_action(self, action):
        """Execute the specified action."""
        if action == 'volume_up':
            return self.volume_up()
        elif action == 'volume_down':
            return self.volume_down()
        elif action == 'brightness_up':
            return self.brightness_up()
        elif action == 'brightness_down':
            return self.brightness_down()
        elif action == 'close_app':
            return 'EXIT'
        return None


def load_model():
    """Load the trained gesture recognition model."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please run 3_train_decision_tree_sklearn.py first!")
        return None
    
    print("Loading gesture recognition model...")
    model_data = joblib.load(MODEL_PATH)
    print(f"Model loaded! Recognizes: {model_data['gesture_names']}")
    return model_data


def extract_landmarks(hand_landmarks):
    """Extract landmark coordinates from MediaPipe hand detection."""
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append(landmark.x)
        landmarks.append(landmark.y)
        landmarks.append(landmark.z)
    return landmarks


def get_gesture_color(gesture_name):
    """Get display color for each gesture."""
    colors = {
        'open_palm': (0, 255, 0),       # Green - Brightness Up
        'fist': (0, 0, 255),            # Red - Brightness Down
        'thumbs_up': (255, 255, 0),     # Cyan - Volume Up
        'thumbs_down': (255, 0, 255),   # Magenta - Volume Down
        'peace_sign': (0, 165, 255)     # Orange - Close App
    }
    return colors.get(gesture_name, (255, 255, 255))


def get_action_text(gesture_name):
    """Get the action description for a gesture."""
    action_texts = {
        'open_palm': 'BRIGHTNESS UP',
        'fist': 'BRIGHTNESS DOWN',
        'thumbs_up': 'VOLUME UP',
        'thumbs_down': 'VOLUME DOWN',
        'peace_sign': 'CLOSE APP'
    }
    return action_texts.get(gesture_name, 'UNKNOWN')


def draw_info(frame, gesture_name, detected, action_feedback=None):
    """Draw information overlay on the video frame."""
    height, width = frame.shape[:2]
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (width - 10, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(frame, "GESTURE WINDOWS CONTROL", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Gesture name and action
    if detected:
        color = get_gesture_color(gesture_name)
        display_text = gesture_name.replace('_', ' ').upper()
        action_text = get_action_text(gesture_name)
        
        cv2.putText(frame, f"Gesture: {display_text}", 
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Action: {action_text}", 
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(frame, "Gesture: NO HAND DETECTED", 
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        cv2.putText(frame, "Action: WAITING...", 
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    # Action feedback
    if action_feedback:
        cv2.putText(frame, action_feedback, 
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Instructions
    cv2.putText(frame, "Press Q to quit", 
                (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Hand detection guide rectangle
    cv2.rectangle(frame, (width//4, height//4), 
                  (3*width//4, 3*height//4), (0, 255, 0), 2)
    
    # Gesture legend (bottom right)
    legend_x = width - 250
    legend_y = height - 150
    legend_items = [
        ("Thumbs Up", "Vol +", (255, 255, 0)),
        ("Thumbs Down", "Vol -", (255, 0, 255)),
        ("Open Palm", "Bright +", (0, 255, 0)),
        ("Fist", "Bright -", (0, 0, 255)),
        ("Peace Sign", "Exit", (0, 165, 255))
    ]
    
    for i, (gesture, action, color) in enumerate(legend_items):
        y_pos = legend_y + (i * 25)
        cv2.putText(frame, f"{gesture}: {action}", 
                    (legend_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    """Main function to run gesture-based Windows control."""
    # Load model
    model_data = load_model()
    if model_data is None:
        return
    
    model = model_data['model']
    idx_to_gesture = model_data['idx_to_gesture']
    
    # Initialize Windows controller
    controller = WindowsController()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return
    
    print("\n" + "=" * 50)
    print("GESTURE WINDOWS CONTROL STARTED")
    print("=" * 50)
    print("\nGesture Mappings:")
    print("  - Thumbs Up    -> Volume Up")
    print("  - Thumbs Down  -> Volume Down")
    print("  - Open Palm    -> Brightness Up")
    print("  - Fist         -> Brightness Down")
    print("  - Peace Sign   -> Close Application")
    print("\nPress Q to quit")
    print("=" * 50 + "\n")
    
    # Gesture stabilization variables
    gesture_history = []
    action_feedback = None
    feedback_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = hands.process(rgb_frame)
        
        gesture_name = "No Hand"
        detected = False
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Extract landmarks and predict
            landmarks = extract_landmarks(hand_landmarks)
            landmarks_array = np.array(landmarks).reshape(1, -1)
            
            prediction = model.predict(landmarks_array)[0]
            gesture_name = idx_to_gesture[prediction]
            detected = True
            
            # Gesture stabilization - require consecutive frames with same gesture
            gesture_history.append(gesture_name)
            if len(gesture_history) > GESTURE_HOLD_FRAMES:
                gesture_history.pop(0)
            
            # Check if gesture is stable
            if len(gesture_history) == GESTURE_HOLD_FRAMES:
                if all(g == gesture_history[0] for g in gesture_history):
                    stable_gesture = gesture_history[0]
                    
                    # Execute action
                    if stable_gesture in GESTURE_ACTIONS:
                        action = GESTURE_ACTIONS[stable_gesture]
                        result = controller.execute_action(action)
                        
                        if result == 'EXIT':
                            print("\nPeace sign detected - Closing application...")
                            break
                        elif result:
                            action_feedback = result
                            feedback_time = time.time()
                    
                    # Clear history after action
                    gesture_history.clear()
        else:
            gesture_history.clear()
        
        # Clear feedback after 2 seconds
        if action_feedback and time.time() - feedback_time > 2:
            action_feedback = None
        
        # Draw info on frame
        draw_info(frame, gesture_name, detected, action_feedback)
        
        # Display
        cv2.imshow('Gesture Windows Control', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed.")


if __name__ == "__main__":
    main()
