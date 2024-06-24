# Running the Script
# For Training: Navigate to the script’s directory in the terminal and run:
# bash
# Copy code
# python your_script_name.py train --dataset path_to_your_dataset
# For Prediction: Use the command:
# bash
# Copy code
# python your_script_name.py predict --new_data path_to_new_data.csv --model path_to_model.h5
# This setup allows you to use the terminal to either train a new model or predict using an existing model by specifying the mode and paths as command-line arguments.

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

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

def clean_data(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    return data

def load_and_preprocess_data(base_path, window_size=50, step_size=25, max_len=100):
    data = []
    labels = []
    for label_folder in os.listdir(base_path):
        label_id = label_folder.split('-')[0]
        label = label_mapping.get(label_id, None)
        if label is None:
            continue
        label_folder_path = os.path.join(base_path, label_folder)
        for volunteer_folder in os.listdir(label_folder_path):
            volunteer_folder_path = os.path.join(label_folder_path, volunteer_folder)
            for test_folder in os.listdir(volunteer_folder_path):
                test_folder_path = os.path.join(volunteer_folder_path, test_folder)
                for sensor_file in os.listdir(test_folder_path):
                    sensor_file_path = os.path.join(test_folder_path, sensor_file)
                    sensor_data = pd.read_csv(sensor_file_path)
                    if sensor_data.empty:
                        continue
                    sensor_data = clean_data(sensor_data)
                    for start_pos in range(0, len(sensor_data) - window_size + 1, step_size):
                        segment = sensor_data.iloc[start_pos:start_pos + window_size]
                        if segment.shape[0] == window_size:
                            data.append(segment.values)
                            labels.append(label)
    data = np.array(data)
    if data.ndim != 3:
        raise ValueError("Data array should have three dimensions (samples, time_steps, features).")
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data)
    data = pad_sequences(data, maxlen=max_len, dtype='float32', padding='post', truncating='post', value=0.0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = data.reshape(-1, data.shape[2])
    data = scaler.fit_transform(data)
    data = data.reshape(-1, max_len, data.shape[1])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return data, labels, scaler, le

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        BatchNormalization(),
        LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dense(len(label_mapping), activation='softmax')
    ])
    return model

def train_and_evaluate_model(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_train1:", X_train.shape[1])
    print("Shape of X_train2:", X_train.shape[2])
    print("Shape of X_test:", X_test.shape)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
    
    # Call to plot accuracy graph
    plot_accuracy(history)
    
    # Load the Label Encoder to get label names for plotting
    le = LabelEncoder()
    le.classes_ = np.array(list(label_mapping.values()))
    
    # Call to plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test, le)
    
    return model, history

def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_and_predict(model_path, new_data):
    model = load_model(model_path)
    prediction = model.predict(new_data)
    return prediction

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("LSTM Accuracy Curve.png")
    plt.show()
    plt.close()

def plot_confusion_matrix(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    labels = le.inverse_transform(range(len(label_mapping)))  # Convert numeric labels back to original string labels
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size for better visibility
    cm_display.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.yticks(rotation=45)
    plt.grid(False)  # Turn off the grid to reduce visual clutter
    plt.savefig("LSTM Confusion Matrix.png")
    plt.show()
    plt.close()

def preprocess_new_data(new_data_path, scaler, max_len=100):
    sensor_data = pd.read_csv(new_data_path)
    sensor_data = sensor_data.values
    if sensor_data.shape[0] < max_len:
        sensor_data = pad_sequences([sensor_data], maxlen=max_len, dtype='float32', padding='post', truncating='post', value=0.0)[0]
    sensor_data = sensor_data.reshape(1, max_len, -1)
    sensor_data = scaler.transform(sensor_data.reshape(-1, sensor_data.shape[2])).reshape(1, max_len, -1)
    return sensor_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Predict using LSTM Model")
    parser.add_argument('mode', choices=['train', 'predict'], help="Mode: train a new model or predict using an existing model")
    parser.add_argument('--dataset', type=str, help="Path to the dataset directory for training")
    parser.add_argument('--new_data', type=str, help="Path to new data for prediction")
    parser.add_argument('--model', type=str, default='lstm_model.h5', help="Path to save or load the model")
    args = parser.parse_args()
    if args.mode == 'train':
        data, labels, scaler, le = load_and_preprocess_data(args.dataset)
        model, history = train_and_evaluate_model(data, labels)
        save_model(model, args.model)
        joblib.dump(scaler, 'scaler.gz')
        joblib.dump(le, 'label_encoder.gz')
        print("Training completed and model saved.")
        final_train_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        print(f"Final Training Accuracy: {final_train_accuracy*100:.2f}%")
        print(f"Final Validation Accuracy: {final_val_accuracy*100:.2f}%")
    elif args.mode == 'predict':
        if not args.new_data:
            print("Error: Please provide a path to the new data for prediction.")
        else:
            scaler = joblib.load('scaler.gz')
            le = joblib.load('label_encoder.gz')
            new_data = preprocess_new_data(args.new_data, scaler)
            prediction = load_and_predict(args.model, new_data)
            predicted_label = le.inverse_transform([np.argmax(prediction)])
            print(f'Predicted Label: {predicted_label[0]}')

