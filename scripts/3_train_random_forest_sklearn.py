import numpy as np
import json
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from sklearn.ensemble import RandomForestClassifier

raw_data_folder = "../data/raw"
augmented_data_folder = "../data/augmented"
model_save_path = "../models/gesture_random_forest_sklearn.pkl"

max_depth = 10
min_samples_split = 10
min_samples_leaf = 5
random_state = 42

gesture_names = ["open_palm", "fist", "thumbs_up", "thumbs_down", "peace_sign"]


def load_gesture_data(use_augmented=True):
    X_list = []
    y_list = []
    
    gesture_to_idx = {}
    for i in range(len(gesture_names)):
        gesture_to_idx[gesture_names[i]] = i
    
    X_raw, y_raw = load_from_folder(raw_data_folder, gesture_to_idx)
    for sample in X_raw:
        X_list.append(sample)
    for label in y_raw:
        y_list.append(label)
    print(f"Raw samples: {len(X_raw)}")
    
    if use_augmented == True:
        if os.path.exists(augmented_data_folder):
            X_aug, y_aug = load_from_folder(augmented_data_folder, gesture_to_idx)
            for sample in X_aug:
                X_list.append(sample)
            for label in y_aug:
                y_list.append(label)
            print(f"Augmented samples: {len(X_aug)}")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Total samples: {len(X)}")
    
    return X, y, gesture_to_idx


def load_from_folder(folder_path, gesture_to_idx):
    X_list = []
    y_list = []
    
    if os.path.exists(folder_path) == False:
        return X_list, y_list
    
    all_items = os.listdir(folder_path)
    person_folders = []
    for item in all_items:
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            person_folders.append(item)
    
    for person in person_folders:
        person_path = os.path.join(folder_path, person)
        all_files = os.listdir(person_path)
        json_files = []
        for file in all_files:
            if file.endswith('.json'):
                json_files.append(file)
        
        for json_file in json_files:
            file_path = os.path.join(person_path, json_file)
            
            f = open(file_path, 'r')
            data = json.load(f)
            f.close()
            
            landmarks = data['landmarks']
            gesture_name = data['gesture']
            
            if gesture_name in gesture_to_idx:
                X_list.append(landmarks)
                label = gesture_to_idx[gesture_name]
                y_list.append(label)
    
    return X_list, y_list


def train_model(X_train, y_train):
    print("TRAINING MODEL")
    
    model = RandomForestClassifier(
        n_estimators=100,           # 100 trees
        criterion='gini',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features='sqrt',        # random subspace per split (stronger decorrelation)
        bootstrap=True,             # random samples with replacement per tree
        oob_score=True,             # out-of-bag validation for a quick generalization check
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"Trees: {len(model.estimators_)}")
    depths = [estimator.get_depth() for estimator in model.estimators_]
    print(f"Avg tree depth: {np.mean(depths):.2f} (min={np.min(depths)}, max={np.max(depths)})")
    if hasattr(model, "oob_score_"):
        print(f"OOB Accuracy: {model.oob_score_ * 100:.2f}%")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, gesture_list):
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracy_percent = train_accuracy * 100
    
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracy_percent = test_accuracy * 100
    
    print(f"Training Accuracy: {train_accuracy_percent:.2f}%")
    print(f"Testing Accuracy: {test_accuracy_percent:.2f}%")
    
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(cm)


def save_model(model, gesture_to_idx):
    model_dir = os.path.dirname(model_save_path)
    if os.path.exists(model_dir) == False:
        os.makedirs(model_dir)
    
    idx_to_gesture = {}
    for name in gesture_to_idx:
        idx = gesture_to_idx[name]
        idx_to_gesture[idx] = name
    
    gesture_list = []
    for i in range(len(gesture_to_idx)):
        gesture_list.append(idx_to_gesture[i])
    
    model_data = {}
    model_data['model'] = model
    model_data['gesture_to_idx'] = gesture_to_idx
    model_data['idx_to_gesture'] = idx_to_gesture
    model_data['gesture_names'] = gesture_list
    
    joblib.dump(model_data, model_save_path)
    full_path = os.path.abspath(model_save_path)
    print(f"\nModel saved: {full_path}")


def main():
    X, y, gesture_to_idx = load_gesture_data(use_augmented=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=random_state, 
                                                         stratify=y)
    
    train_size = len(X_train)
    test_size = len(X_test)
    print(f"\nTrain: {train_size}, Test: {test_size}")
    
    model = train_model(X_train, y_train)
    
    idx_to_gesture = {}
    for name in gesture_to_idx:
        idx = gesture_to_idx[name]
        idx_to_gesture[idx] = name
    
    gesture_list = []
    for i in range(len(gesture_to_idx)):
        gesture_list.append(idx_to_gesture[i])
    
    evaluate_model(model, X_train, y_train, X_test, y_test, gesture_list)
    
    save_model(model, gesture_to_idx)
    
    print("\nTraining complete")


if __name__ == "__main__":
    main()
