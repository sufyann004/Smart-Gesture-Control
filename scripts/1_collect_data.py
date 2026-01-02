import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

GESTURES = [
    "open_palm",      
    "fist",           
    "thumbs_up",      
    "thumbs_down",    
    "peace_sign"     
]

SAMPLES_PER_GESTURE = 40

DATA_FOLDER = "../data/raw"

mp_hands = mp.solutions.hands # MediaPipe hands module
mp_drawing = mp.solutions.drawing_utils # MediaPipe drawing module
hands = mp_hands.Hands(
    max_num_hands=1, # We only need to track one hand
    min_detection_confidence=0.7, # Minimum confidence for initial detection
    min_tracking_confidence=0.5 # Minimum confidence for tracking the hand
)

def extract_landmarks(hand_landmarks):
    landmarks = []

    for landmark in hand_landmarks.landmark:
        
        landmarks.append(landmark.x)  # Horizontal position (0 to 1)
        landmarks.append(landmark.y)  # Vertical position (0 to 1)
        landmarks.append(landmark.z)  # Depth position (negative values = closer)
    
    return landmarks


def save_sample(landmarks, gesture_name, person_name, sample_number):
    
    person_folder = os.path.join(DATA_FOLDER, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    
    # Add timestamp to filename 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    data = {
        "gesture": gesture_name,
        "person": person_name,
        "sample_number": sample_number,
        "landmarks": landmarks,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_landmarks": len(landmarks)
    }
    filename = f"{person_name}_{gesture_name}_{sample_number:03d}_{timestamp}.json"
    filepath = os.path.join(person_folder, filename)
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved: {filename}")


def draw_info_panel(frame, gesture_name, samples_collected, total_samples):
   
    height, width = frame.shape[:2]

    
    cv2.putText(frame, f"Gesture: {gesture_name.replace('_', ' ').upper()}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Progress: {samples_collected}/{total_samples}", 
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, "Press SPACE to record | Press Q to quit", 
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Progress bar
    bar_width = int((samples_collected / total_samples) * (width - 40))
    cv2.rectangle(frame, (20, 125), (20 + bar_width, 140), (0, 255, 0), -1)
    cv2.rectangle(frame, (20, 125), (width - 20, 140), (255, 255, 255), 2)
    
    #  hand detection area guide
    cv2.rectangle(frame, (width//4, height//4), 
                  (3*width//4, 3*height//4), (0, 255, 0), 2)




if __name__ == "__main__":

    person_name = input("\nEnter your name").strip().lower()
    
    if not person_name:
        print("ERROR: Name cannot be empty!")
        exit()
       
    print("Which gesture do you want to record?")
    for i, gesture in enumerate(GESTURES, 1):
        print(f"  {i}. {gesture.replace('_', ' ').title()}")
    

    while True:
        try:
            choice = input(f"\nEnter number (1-5): ").strip()
            choice_number = int(choice)
            
            if 1 <= choice_number <= len(GESTURES):
                selected_gesture = GESTURES[choice_number - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(GESTURES)}")
        except:
            print("Please enter a valid number!")
    
    # Checking if data folder exists
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f" Created data folder: {DATA_FOLDER}")
    
    # Create person folder if needed
    person_folder = os.path.join(DATA_FOLDER, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
        print(f" Created folder for {person_name}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\nERROR: Could not open webcam!")
        exit()
    
   
    print("Keep your hand inside the green rectangle")
    print("Try different angles and distances")
    print("Press SPACE when you see 'HAND DETECTED' to record")
    print("Press Q to quit anytime")
    print("\nPress any key to start...")
    cv2.waitKey(0)
    # Counter for samples
    samples_collected = 0
    # Main loop - collect samples
    while samples_collected < SAMPLES_PER_GESTURE:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Failed to read from webcam")
            break
        
        # Flip frame horizontally (mirror view - easier to use)
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = hands.process(rgb_frame)
        
        # Draw info panel
        draw_info_panel(frame, selected_gesture, samples_collected, SAMPLES_PER_GESTURE)
        
        # Check if hand is detected
        if results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Show message
            cv2.putText(frame, "HAND DETECTED - Press SPACE to record!", 
                       (20, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # No hand detected
            cv2.putText(frame, "NO HAND DETECTED ", 
                       (20, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Gesture Data Collection', frame)
        
        # Wait for key press (1 millisecond)
        key = cv2.waitKey(1) & 0xFF
        
        # If SPACE is pressed
        if key == ord(' '):
            if results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = extract_landmarks(results.multi_hand_landmarks[0])
                
                # Save the sample
                samples_collected = samples_collected + 1
                save_sample(landmarks, selected_gesture, person_name, samples_collected)
                
                # Show visual feedback
                cv2.rectangle(frame, (0, 0), 
                            (frame.shape[1], frame.shape[0]), 
                            (0, 255, 0), 10)
                cv2.putText(frame, "RECORDED!", 
                          (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.imshow('Gesture Data Collection', frame)
                cv2.waitKey(200)  # Wait 200 milliseconds
                
                print(f"Sample {samples_collected}/{SAMPLES_PER_GESTURE} recorded!")
            else:
                print("No hand detected!")
        
        # If Q is pressed, quit
        elif key == ord('q'):
            print("\n Quitting")
            break
    
    # Close webcam and windows
    cap.release()
    cv2.destroyAllWindows()
    
   
    if samples_collected == SAMPLES_PER_GESTURE:
        print(f"Successfully recorded {samples_collected} samples of '{selected_gesture}'")
    else:
        print(f"Recording stopped. Collected {samples_collected} samples.")
    print(f"Data saved in: {os.path.abspath(person_folder)}/")
   
