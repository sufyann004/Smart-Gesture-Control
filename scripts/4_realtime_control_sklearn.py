import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

MODEL_PATH = "../models/gesture_random_forest_sklearn.pkl"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


def load_model():

    print("Loading model...")
    model_data = joblib.load(MODEL_PATH)
    print(f"Model loaded! Recognizes: {model_data['gesture_names']}")
    return model_data


def extract_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append(landmark.x)
        landmarks.append(landmark.y)
        landmarks.append(landmark.z)
    return landmarks


def get_gesture_color(gesture_name):
    colors = {
        'open_palm': (0, 255, 0),       # Green
        'fist': (0, 0, 255),            # Red
        'thumbs_up': (255, 255, 0),     # Cyan
        'thumbs_down': (255, 0, 255),   # Magenta
        'peace_sign': (0, 165, 255)     # Orange
    }
    return colors.get(gesture_name, (255, 255, 255))


def draw_info(frame, gesture_name, detected):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Gesture name
    if detected:
        color = get_gesture_color(gesture_name)
        display_text = gesture_name.replace('_', ' ').upper()
    else:
        color = (100, 100, 100)
        display_text = "NO HAND DETECTED"
    
    cv2.putText(frame, f"Gesture: {display_text}", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    cv2.putText(frame, "Press Q to quit", 
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Hand detection guide
    cv2.rectangle(frame, (width//4, height//4), 
                  (3*width//4, 3*height//4), (0, 255, 0), 2)


def main():
    model_data = load_model()
    if model_data is None:
        return
    
    model = model_data['model']
    idx_to_gesture = model_data['idx_to_gesture']
    
    # Open webcam
    cap = cv2.VideoCapture(0)
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
        
        # Draw info on frame
        draw_info(frame, gesture_name, detected)
        
        # Display
        cv2.imshow('Gesture Recognition - Sklearn', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nProgram ended.")


if __name__ == "__main__":
    main()
