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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
                    # Ensure the CSV has data and avoid empty data frames
                    if sensor_data.empty:
                        continue
                    # Collect segments of sensor data
                    for start_pos in range(0, len(sensor_data) - window_size + 1, step_size):
                        segment = sensor_data.iloc[start_pos:start_pos + window_size]
                        if segment.shape[0] == window_size:
                            data.append(segment.values)  # Store as array
                            labels.append(label)

    # Convert list to numpy array
    data = np.array(data)
    if data.ndim != 3:
        raise ValueError("Data array should have three dimensions (samples, time_steps, features).")
    if np.isnan(data).any() or np.isinf(data).any():
        print("Data contains NaN or infinite values. Replacing with zeros.")
        data = np.nan_to_num(data)

    # Pad data to ensure all sequences have the same length
    data = pad_sequences(data, maxlen=max_len, dtype='float32', padding='post', truncating='post', value=0.0)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = data.reshape(-1, data.shape[2])  # Flatten to fit scaler (assuring 3D to 2D conversion)
    data = scaler.fit_transform(data)
    data = data.reshape(-1, max_len, data.shape[1])  # Reshape back to 3D after normalization

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    return data, labels, scaler, le

def build_cnn_model(input_shape):
    model = Sequential()
    # Adjust the number of input channels (features per timestep) in the first Conv1D layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Additional layers
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))  # Assuming 10 classes

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(data, labels, le):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Calculate the number of features
    num_features = X_train.shape[1]  # Assuming the number of features is the second dimension of X_train
    
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)

    # # Correctly reshape the data for a 1D CNN, assuming each feature is a separate channel
    # X_train = X_train.reshape((X_train.shape[0], num_features, 1))
    # X_test = X_test.reshape((X_test.shape[0], num_features, 1))
    
    # Build and train the model
    model = build_cnn_model((num_features, 10))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    
    # Print final accuracies for training and validation sets
    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
    print(f"Final Validation Accuracy: {final_val_accuracy * 100:.2f}%")
    
    # Optional: Plot accuracy and confusion matrix
    plot_accuracy(history)
    plot_confusion_matrix(model, X_test, y_test, le)
    
    return model, history

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("CNN_V1 Accuracy Curve.png")
    plt.show()
    plt.close

def plot_confusion_matrix(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    labels = le.inverse_transform(range(len(label_mapping)))  # Convert numeric labels back to original string labels
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size for better visibility
    cm_display.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('CNN_V1 Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.yticks(rotation=45)
    plt.grid(False)  # Turn off the grid to reduce visual clutter
    plt.savefig("CNN Confusion Matrix.png")
    plt.show()
    plt.close()


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
    parser = argparse.ArgumentParser(description="Train and Predict using CNN Model")
    parser.add_argument('mode', choices=['train', 'predict'], help="Mode: train a new model or predict using an existing model")
    parser.add_argument('--dataset', type=str, help="Path to the dataset directory for training")
    parser.add_argument('--new_data', type=str, help="Path to new data for prediction")
    parser.add_argument('--model', type=str, default='cnn_model.h5', help="Path to save or load the model")

    args = parser.parse_args()

    if args.mode == 'train':
        data, labels, scaler, le = load_and_preprocess_data(args.dataset)
        model, history = train_model(data, labels,le)
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

