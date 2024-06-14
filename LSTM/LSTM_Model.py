import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam  # Updated optimizer import
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from glob import glob

# Ensure the latest features are being used (if compatible with your TensorFlow version)
tf.compat.v1.disable_eager_execution()  # Disable eager execution if needed

def load_data_from_directory(directory_path):
    file_paths = glob(os.path.join(directory_path, '**/*.csv'), recursive=True)
    return file_paths

def preprocess_data(file_path, window_size, step_size):
    data = pd.read_csv(file_path)
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data.iloc[:, 1:10])
    windows = []
    for start in range(0, normalized_data.shape[0] - window_size + 1, step_size):
        end = start + window_size
        window = normalized_data[start:end, :]
        if window.shape[0] == window_size:
            windows.append(window)
    return windows

def organize_data(file_paths, window_size, step_size):
    label_data = {}
    for file_path in file_paths:
        label = os.path.basename(os.path.dirname(file_path))
        if label not in label_data:
            label_data[label] = []
        data_windows = preprocess_data(file_path, window_size, step_size)
        label_data[label].extend(data_windows)
    return label_data

def build_and_train_model(data_by_label):
    labels = sorted(data_by_label.keys())
    label_dict = {label: i for i, label in enumerate(labels)}
    X = []
    y = []
    for label, windows in data_by_label.items():
        for window in windows:
            X.append(window)
            y.append(label_dict[label])

    X = np.array(X)
    y = np.array(y)

    # Pad sequences
    X_padded = pad_sequences(X, padding='post', dtype='float32')
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
    print(f"{y_test.shape=}")

    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.5),
        LSTM(20),
        Dropout(0.5),
        Dense(len(labels), activation='softmax')
    ])

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Check and clean data before training
    X_train = np.nan_to_num(X_train)
    y_train = np.nan_to_num(y_train)

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)
    
    print(model.summary())

    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    if np.isnan(predictions).any():
        print("NaN found in predictions")
        predictions = np.nan_to_num(predictions)

    # Assuming predictions are probabilities and y_test is already class indices
    predicted_classes = np.argmax(predictions, axis=1)
    if y_test.ndim == 1:
        true_classes = y_test  # If y_test is simply a 1D array of class indices
    else:
        true_classes = np.argmax(y_test, axis=1)  # If y_test is one-hot encoded



    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:")
    print(cm)

 # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, predicted_classes, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Training accuracy chart
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

# Command line arguments handling
if len(sys.argv) < 2:
    print("Usage: python script_name.py <directory_path>")
    sys.exit(1)

directory_path = sys.argv[1]
window_size = 25
step_size = 12

file_paths = load_data_from_directory(directory_path)
data_by_label = organize_data(file_paths, window_size, step_size)
build_and_train_model(data_by_label)
