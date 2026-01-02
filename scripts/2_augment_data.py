import numpy as np
import json
import os
from datetime import datetime

RAW_DATA_FOLDER = "../data/raw"
AUGMENTED_DATA_FOLDER = "../data/augmented"

# Augmentation settings
ROTATION_ANGLES = [-15, -10, -5, 5, 10, 15]  # Rotate by these degrees
SCALE_FACTORS = [0.9, 1.1]                    # Scale by 90% and 110%
TRANSLATION_RANGE = 0.05                       # Shift by up to 5%
NOISE_LEVEL = 0.01                             # Add 1% random noise

def reshape_landmarks(landmarks):
   # convert flat list to nested list 
    return np.array(landmarks).reshape(21, 3)


def flatten_landmarks(landmarks):
  # convert 21 nested lists to flat list of 63 points
    return landmarks.flatten().tolist()


def rotate_landmarks(landmarks, angle_degrees):
  #Rotate hand landmarks around the center point.
    # Convert angle to radians 
    angle_rad = np.radians(angle_degrees)
    
    # Find the center of the hand 
    center_x = np.mean(landmarks[:, 0])
    center_y = np.mean(landmarks[:, 1])
    
    # Create rotation matrix
    # This is basic 2D rotation: [cos(θ), -sin(θ)]
    #                            [sin(θ),  cos(θ)]
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Copy landmarks so we don't modify original
    rotated = landmarks.copy()
    
    # Rotate each point around the center
    for i in range(21):
        # Translate point to origin (subtract center)
        x = landmarks[i, 0] - center_x
        y = landmarks[i, 1] - center_y
        
        #  rotation
        rotated[i, 0] = x * cos_angle - y * sin_angle + center_x
        rotated[i, 1] = x * sin_angle + y * cos_angle + center_y
        # Keep z coordinate unchanged
        rotated[i, 2] = landmarks[i, 2]
    
    return rotated


def scale_landmarks(landmarks, scale_factor):
   # Scale hand landmarks to make hand appear larger or smaller.
    
    # Find the center of the hand
    center = np.mean(landmarks, axis=0)
    
    # Scale each point relative to center
    scaled = center + scale_factor * (landmarks - center)
    
    return scaled


def translate_landmarks(landmarks, max_translation):
       # Shift hand landmarks in random direction.
        
   
    # Generate random shift for x, y, z
    shift_x = np.random.uniform(-max_translation, max_translation)
    shift_y = np.random.uniform(-max_translation, max_translation)
    shift_z = np.random.uniform(-max_translation, max_translation)
    
    # Apply shift to all points
    translated = landmarks.copy()
    translated[:, 0] += shift_x
    translated[:, 1] += shift_y
    translated[:, 2] += shift_z
    
    return translated


def add_noise(landmarks, noise_level):
    
    
    #This simulates imperfect hand detection and makes model more robust.
    
   
    # Generate random noise with same shape as landmarks
    noise = np.random.normal(0, noise_level, landmarks.shape)
    
    # Add noise to landmarks
    noisy = landmarks + noise
    
    return noisy



def augment_sample(landmarks_flat):
 
    augmented_samples = []
    
    # Convert to 21×3 array for easier manipulation
    landmarks = reshape_landmarks(landmarks_flat)
    
    # 1. ROTATION AUGMENTATION
    
    for angle in ROTATION_ANGLES:
        rotated = rotate_landmarks(landmarks, angle)
        augmented_samples.append(flatten_landmarks(rotated))
    
    # 2. SCALING AUGMENTATION
    
    for scale in SCALE_FACTORS:
        scaled = scale_landmarks(landmarks, scale)
        augmented_samples.append(flatten_landmarks(scaled))
    
    # 3. TRANSLATION AUGMENTATION
    
    translated = translate_landmarks(landmarks, TRANSLATION_RANGE)
    augmented_samples.append(flatten_landmarks(translated))
    
    # 4. NOISE AUGMENTATION
 
    noisy = add_noise(landmarks, NOISE_LEVEL)
    augmented_samples.append(flatten_landmarks(noisy))
    
    return augmented_samples


def load_sample(filepath):
   
    with open(filepath, 'r') as f:
        return json.load(f)


def save_augmented_sample(data, person_name, gesture_name, 
                         original_sample_num, aug_num):

    # Create folder for this person if needed
    person_folder = os.path.join(AUGMENTED_DATA_FOLDER, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    
    # Create filename with timestamp to prevent overwriting previous augmentations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{person_name}_{gesture_name}_{original_sample_num:03d}_aug_{aug_num:02d}_{timestamp}.json"
    filepath = os.path.join(person_folder, filename)
    
    # Add augmentation info to data
    data['is_augmented'] = True
    data['augmentation_number'] = aug_num
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def process_all_data():
    
    # Create augmented data folder if needed
    if not os.path.exists(AUGMENTED_DATA_FOLDER):
        os.makedirs(AUGMENTED_DATA_FOLDER)
        print(f"Created augmented data folder: {AUGMENTED_DATA_FOLDER}")
    
    total_files = 0
    processed_files = 0
    total_augmented = 0
    
    # Get list of all person folders
    person_folders = [f for f in os.listdir(RAW_DATA_FOLDER) 
                     if os.path.isdir(os.path.join(RAW_DATA_FOLDER, f))]
    
    if not person_folders:
        print("ERROR: No data found in raw data folder!")
        print(f"Expected to find person folders in: {RAW_DATA_FOLDER}")
        return
    
    print(f"\nFound data from {len(person_folders)} people: {', '.join(person_folders)}")
    print("\nStarting augmentation process...")
    print("-" * 60)
    
    # Process each person's data
    for person_name in person_folders:
        person_path = os.path.join(RAW_DATA_FOLDER, person_name)
        
        print(f"\n Processing data for: {person_name}")
        
        # Get all JSON files for this person
        json_files = [f for f in os.listdir(person_path) if f.endswith('.json')]
        total_files += len(json_files)
        
        print(f"   Found {len(json_files)} samples")
        
        # Process each sample
        for json_file in json_files:
            filepath = os.path.join(person_path, json_file)
            
            # Load original sample
            sample_data = load_sample(filepath)
            
            # Extract info from filename 
            parts = json_file.replace('.json', '').split('_')
            gesture_name = '_'.join(parts[1:-1])  # Handle multi-word gestures
            sample_num = int(parts[-1])
            
            # Create augmented versions
            augmented_landmarks_list = augment_sample(sample_data['landmarks'])
            
            # Save each augmented version
            for aug_idx, aug_landmarks in enumerate(augmented_landmarks_list, 1):
                # Create new data dictionary
                aug_data = sample_data.copy()
                aug_data['landmarks'] = aug_landmarks
                aug_data['original_sample'] = json_file
                
                # Save augmented sample
                save_augmented_sample(aug_data, person_name, gesture_name, 
                                    sample_num, aug_idx)
                total_augmented += 1
            
            processed_files += 1
            
            # Show progress every 20 files
            if processed_files % 20 == 0:
                print(f"   Progress: {processed_files}/{total_files} samples processed...")
        
        print(f"   Completed {person_name}: {len(json_files)} original → {len(json_files) * 10} augmented")
    
    # Final summary
   
    
    print(f"Original samples: {total_files}")
    print(f"Augmented samples created: {total_augmented}")
    print(f"Total dataset size: {total_files + total_augmented}")
    print(f"\nAugmentation ratio: {total_augmented / total_files:.1f}x")
    print(f"\nAugmented data saved in: {os.path.abspath(AUGMENTED_DATA_FOLDER)}")
    



if __name__ == "__main__":
   
    process_all_data()
    
  
    
