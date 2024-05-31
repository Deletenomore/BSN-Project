import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, TimeDistributed
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import joblib

# Define the label mapping
label_mapping = {
    '901': 'front-sitting', '902': 'front-protecting', '903': 'front-knees',
    '904': 'front-knees-lying', '905': 'front-quick-recovery', '906': 'front-slow-recovery',
    '907': 'front-right', '908': 'front-left', '909': 'back-sitting', '910': 'back-knees'
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
                combined_data = np.concatenate(combined_data)
                data.append(combined_data)
                labels.append(label)
    data = np.array(data)
    labels = np.array(labels)

    # Data normalization
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    return data, labels, scaler, le

def build_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Assuming 10 classes
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    n_seq = 2
    n_steps = X_train.shape[1] // n_seq
    n_features = 1
    X_train = X_train.reshape((X_train.shape[0], n_seq, n_steps, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_seq, n_steps, n_features))
    model = build_cnn_lstm_model((n_seq, n_steps, n_features))
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

# Loading the model and making predictions
def load_and_predict(model_path, new_data):
    model = load_model(model_path)
    prediction = model.predict(new_data)
    return prediction

# Preprocess new data (Example)
def preprocess_new_data(new_data_path, scaler):
    # Assuming new_data_path is a path to a CSV file containing new data
    sensor_data = pd.read_csv(new_data_path)
    sensor_data = sensor_data.values.flatten()
    sensor_data = sensor_data.reshape(1, -1)  # Reshape if necessary to fit model input
    sensor_data = scaler.transform(sensor_data)  # Scale data
    return sensor_data

if __name__ == "__main__":
    # Load data, train model, etc.
    base_path = "path_to_your_data"
    data, labels, scaler, le = load_and_preprocess_data(base_path)
    model, history = train_model(data, labels)

    # Save the model
    save_model(model, 'cnn_lstm_model.h5')

    # Example: Load the model and predict new data
    new_data_path = "path_to_new_data.csv"
    new_data = preprocess_new_data(new_data_path, scaler)
    prediction = load_and_predict('cnn_lstm_model.h5', new_data)
    print("Prediction:", prediction)