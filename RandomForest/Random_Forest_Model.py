
# To run the provided Python script (Random_Forest_Model.py) from the command line and utilize its functionalities for training or predicting, follow these steps to create a comprehensive guide:

# Requirements
# Ensure Python 3 and pip are installed on your system.
# Install necessary Python packages using pip:
# pip install numpy pandas scikit-learn matplotlib seaborn joblib

# Step-by-Step Instructions
# 1. Training the Model
# To train the RandomForest model using the script, you need to provide the dataset path. The model and its components will be saved to specified paths.

# Command Structure:

# python random_forest.py train --dataset <path_to_training_data>
# Example Command:
# python random_forest.py train --dataset "BSN-Project/Data/Training"
# This command will train the model using data from the specified directory and save the trained model and its components (scaler, label encoder, and imputer) in the default or specified paths.

# 2. Predicting Using the Model
# To make predictions using the trained model, you must specify the path to new data.

# Command Structure:
# python random_forest.py predict --new_data <path_to_new_data>
# Example Command:
# python random_forest.py predict --new_data "/BSN-Project/Data/newPredict"
# This command will use the pre-trained model to make predictions on the new data provided in the specified directory. It assumes that the model and its components are saved in the default paths set in the script.

# Optional Arguments
# The script also includes options to specify paths for the model and its components:

# --model <path_to_model>: Path where the model is saved or should be loaded from.
# --scaler <path_to_scaler>: Path for the scaler file.
# --label_encoder <path_to_label_encoder>: Path for the label encoder file.
# --imputer <path_to_imputer>: Path for the imputer file.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,auc,roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
import joblib
import argparse
import seaborn as sns
from itertools import cycle

# Define the label mapping
label_mapping = {
    '901': 'front-lying',
    '902': 'front-protecting-lying',
    '903': 'front-knees',
    '904': 'front-knees-lying',
    '905': 'front-quick-recovery',
    '906': 'front-slow-recovery',
    '907': 'front-right',
    '908': 'front-left',
    '909': 'back-sitting',
    '910': 'back-lying'
}

def load_data(base_path):
    data = []
    labels = []
    for label_folder in os.listdir(base_path):
        label_id = label_folder.split('-')[0]
        label = label_mapping.get(label_id)
        if label is None:
            continue
        label_folder_path = os.path.join(base_path, label_folder)
        for volunteer_folder in os.listdir(label_folder_path):
            volunteer_folder_path = os.path.join(label_folder_path, volunteer_folder)
            for test_folder in os.listdir(volunteer_folder_path):
                test_folder_path = os.path.join(volunteer_folder_path, test_folder)
                combined_data = []
                min_length = None
                for sensor_file in os.listdir(test_folder_path):
                    sensor_file_path = os.path.join(test_folder_path, sensor_file)
                    sensor_data = pd.read_csv(sensor_file_path)
                    if min_length is None or len(sensor_data) < min_length:
                        min_length = len(sensor_data)
                    combined_data.append(sensor_data.values)
                combined_data = [arr[:min_length] for arr in combined_data]
                combined_data = np.concatenate(combined_data, axis=1)
                data.append(combined_data)
                labels.append(label)
    max_length = max(len(d) for d in data)
    data = [np.pad(d, ((0, max_length - len(d)), (0, 0)), 'constant') for d in data]
    return np.array(data), np.array(labels)

def preprocess_data(base_path):
    data, labels = load_data(base_path)
    data = data.reshape(data.shape[0], -1)
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    classes = np.unique(labels)
    return X_train, X_test, y_train, y_test, scaler, le, classes,imputer

def train_and_evaluate(X_train, X_test, y_train, y_test, classes,le):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_score = model.predict_proba(X_test)  # Get class probabilities for ROC
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
     # Plot ROC Curve and Confusion Matrix
    n_classes = len(le.classes_)
    plot_roc_curve(y_test, y_score, n_classes, le)
    plot_confusion_matrix(y_test, y_pred, [le.inverse_transform([i])[0] for i in range(n_classes)])

    return model

def plot_roc_curve(y_test, y_score, n_classes, le):
    # Binarize the output
    y_test = label_binarize(y_test, classes=[i for i in range(n_classes)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'gray'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(le.inverse_transform([i])[0], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("Random_Forest_ROC")
    plt.close() 

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("Random_Forest_CM")
    plt.close() 

def save_model(model, scaler, label_encoder, imputer, model_path, scaler_path, label_encoder_path, imputer_path):
    """
    Saves the model and its associated preprocessing components to disk.

    Args:
    model (RandomForestClassifier): Trained model to be saved.
    scaler (StandardScaler): Scaler used for data normalization.
    label_encoder (LabelEncoder): Encoder used for transforming labels.
    imputer (SimpleImputer): Imputer used for filling missing values.
    model_path (str): File path to save the trained model.
    scaler_path (str): File path to save the scaler.
    label_encoder_path (str): File path to save the label encoder.
    imputer_path (str): File path to save the imputer.
    """
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(imputer, imputer_path)
    print(f"Model components saved successfully:\nModel: {model_path}\nScaler: {scaler_path}\nLabel Encoder: {label_encoder_path}\nImputer: {imputer_path}")


def preprocess_new_data(new_data_path, scaler, imputer):
    try:
        # Load new data and store each sensor data's flattened array
        new_data = []
        min_length = None
        for sensor_file in os.listdir(new_data_path):
            sensor_file_path = os.path.join(new_data_path, sensor_file)
            sensor_data = pd.read_csv(sensor_file_path)
            if min_length is None or len(sensor_data) < min_length:
                min_length = len(sensor_data)
            new_data.append(sensor_data.values)

        # Normalize the length of all sensor data arrays
        new_data = [arr[:min_length] for arr in new_data]
        new_data = np.concatenate(new_data, axis=1)

        # Flatten the data and reshape for compatibility with scaler and imputer
        total_features = imputer.n_features_in_
        new_data = new_data.flatten()
        if new_data.shape[0] < total_features:
            new_data = np.pad(new_data, (0, total_features - new_data.shape[0]), 'constant')
        elif new_data.shape[0] > total_features:
            new_data = new_data[:total_features]

        new_data = new_data.reshape(1, -1)

        # Impute and scale the data
        new_data = imputer.transform(new_data)
        new_data_scaled = scaler.transform(new_data)

        return new_data_scaled
    except Exception as e:
        print(f"An error occurred during preprocessing: {str(e)}")
        raise


def predict_new_data(new_data_path, model_path, scaler_path, le_path, imputer_path):
    try:
        rForest_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        imputer = joblib.load(imputer_path)

        new_data_scaled = preprocess_new_data(new_data_path, scaler, imputer)
        prediction = rForest_model.predict(new_data_scaled)
        predicted_label = le.inverse_transform(prediction)

        return predicted_label
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Predict using RandomForest model for data classification")
    parser.add_argument('mode', choices=['train', 'predict'], help="Mode: train a new model or predict using an existing model")
    parser.add_argument('--dataset', type=str, help="Path to the dataset directory for training")
    parser.add_argument('--new_data', type=str, help="Path to the new data directory for prediction")
    parser.add_argument('--model', type=str, default='trained_model/model.joblib', help="Path to save or load the model")
    parser.add_argument('--scaler', type=str, default='trained_model/scaler.joblib', help="Path to save or load the scaler")
    parser.add_argument('--label_encoder', type=str, default='trained_model/label_encoder.joblib', help="Path to save or load the label encoder")
    parser.add_argument('--imputer', type=str, default='trained_model/imputer.joblib', help="Path to save or load the imputer")

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.dataset:
            print("Please provide the path to the dataset using --dataset")
            exit(1)
        X_train, X_test, y_train, y_test, scaler, le, classes, imputer = preprocess_data(args.dataset)
        model = train_and_evaluate(X_train, X_test, y_train, y_test,classes,le)
        save_model(model, scaler, le, imputer, args.model, args.scaler, args.label_encoder, args.imputer)
        print(f'Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}')
    elif args.mode == 'predict':
        if not args.new_data:
            print("Please provide the path to the new data using --new_data")
            exit(1)
        prediction = predict_new_data(args.new_data, args.model, args.scaler, args.label_encoder, args.imputer)
        print(f'Predicted Fall Action Label: {prediction[0]}')


