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
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import argparse


# Define the label mapping based on your provided labels
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

def load_and_preprocess_data(base_path):
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
                for sensor_file in os.listdir(test_folder_path):
                    sensor_file_path = os.path.join(test_folder_path, sensor_file)
                    sensor_data = pd.read_csv(sensor_file_path)
                    combined_data.append(sensor_data.values.flatten())

                # Find the max length of sequences in the current dataset or set a fixed length
                max_length = 500  # Adjust based on dataset analysis
                standardized_data = [np.pad(arr, (0, max_length - len(arr)), mode='constant') if len(arr) < max_length else arr[:max_length] for arr in combined_data]
                combined_data = np.concatenate(standardized_data)
                data.append(combined_data)
                labels.append(label)

    data = np.array(data)
    if np.isnan(data).any() or np.isinf(data).any():
        print("Data contains NaN or infinite values. Replacing with zeros.")
        data = np.nan_to_num(data)

    labels = np.array(labels)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return data, labels, scaler, le

def build_cnn_model(input_shape):
    model = Sequential()
    # First convolutional layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Second convolutional layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))  # Assuming 10 classes for classification

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # Ensure data is in the correct shape for a 1D CNN (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Reshape for CNN, assuming each feature is a separate channel
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = build_cnn_model((X_train.shape[1], 1))  # Input shape is (time steps, features)
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    plot_accuracy(history)
    return model, history

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# Saving the model
def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved to {model_path}")

def preprocess_new_data(new_data_path, scaler):
    # Assuming new_data_path is a path to a CSV file containing new data
    sensor_data = pd.read_csv(new_data_path)
    sensor_data = sensor_data.values.flatten()
    sensor_data = sensor_data.reshape(1, -1)  # Reshape if necessary to fit model input
    sensor_data = scaler.transform(sensor_data)  # Scale data
    return sensor_data

def load_and_predict(model_path, new_data):
    model = load_model(model_path)
    prediction = model.predict(new_data)
    return prediction



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and Predict using CNN Model")
    parser.add_argument('mode', choices=['train', 'predict'], help="Mode: train a new model or predict using an existing model")
    parser.add_argument('--dataset', type=str, help="Path to the dataset directory for training")
    parser.add_argument('--new_data', type=str, help="Path to new data for prediction")
    parser.add_argument('--model', type=str, default='cnn_model.h5', help="Path to save or load the model")

    args = parser.parse_args()

    if args.mode == 'train':
        data, labels, scaler, le = load_and_preprocess_data(args.dataset)
        model, history = train_model(data, labels)
        save_model(model, args.model)
        # Optionally save the scaler and label encoder for later use in prediction
        joblib.dump(scaler, 'scaler.gz')
        joblib.dump(le, 'label_encoder.gz')
        print("Training completed and model saved.")
    elif args.mode == 'predict':
        if not args.new_data:
            print("Error: Please provide a path to the new data for prediction.")
        else:
            # Load scaler and label encoder
            scaler = joblib.load('scaler.gz')
            le = joblib.load('label_encoder.gz')
            prediction = load_and_predict(args.model, args.new_data, scaler)
            predicted_label = le.inverse_transform([np.argmax(prediction)])
            print(f'Predicted Label: {predicted_label[0]}')

