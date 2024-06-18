import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from itertools import cycle

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

def calculate_features(segment):
    N = len(segment)
    mu = np.mean(segment)
    sigma = np.std(segment)
    # Manual calculations of skewness and kurtosis
    skewness = (1 / (N * sigma**3)) * np.sum((segment - mu)**3)
    kurtosis = (1 / (N * sigma**4)) * np.sum((segment - mu)**4)

    features = {
        'min': np.min(segment),
        'max': np.max(segment),
        'mean': mu,
        'variance': np.var(segment, ddof=0),
        'skewness': skewness,
        'kurtosis': kurtosis,
        
    }

    autocorrs = np.correlate(segment - mu, segment - mu, mode='full')[N-1:N+10] / (N - np.arange(0, 11))
    features.update({f'autocorr_{delta}': autocorrs[delta] for delta in range(11)})
    
    dft = np.fft.fft(segment)
    spectrum = np.abs(dft)
    frequencies = np.fft.fftfreq(N, d=1/25)
    peaks, _ = find_peaks(spectrum)
    top_peaks = peaks[:5] if len(peaks) > 5 else peaks
    
    features.update({f'peak_{i}': spectrum[top_peak] for i, top_peak in enumerate(top_peaks)})
    features.update({f'freq_{i}': frequencies[top_peak] for i, top_peak in enumerate(top_peaks)})
    
    return features

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
                combined_features = []
                for sensor_file in os.listdir(test_folder_path):
                    sensor_file_path = os.path.join(test_folder_path, sensor_file)
                    sensor_data = pd.read_csv(sensor_file_path)
                    feature_vector = []
                    for column in sensor_data.columns:
                        if column != 'Time':
                            features = calculate_features(sensor_data[column])
                            feature_vector.extend(list(features.values()))
                    combined_features.extend(feature_vector)

                if combined_features:
                    # Check the length of combined_features and correct if not uniform
                    expected_length = 100  # Define the expected length based on your feature extraction logic
                    if len(combined_features) != expected_length:
                        #print(f"Adjusting feature length in {test_folder_path}, original length: {len(combined_features)}")
                        combined_features = combined_features[:expected_length] + [0] * (expected_length - len(combined_features))

                    data.append(combined_features)
                    labels.append(label)
                else:
                    print(f"No features extracted for {test_folder_path}")

    if not data:
        print("No data could be collected, please check the dataset and paths.")
        return None  # Or handle this case as needed

    data = np.array(data, dtype=float)
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data)

    labels = np.array(labels)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return data, labels, scaler, le



def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_roc_curve(y_test, y_score, num_classes, labels):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test = label_binarize(y_test, classes=[i for i in range(num_classes)])
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Debugging: Check the size of the labels array
    #print(f"Number of labels provided: {len(labels)}; Expected: {num_classes}")

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for a specific class
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CNN With Feature Selection ROC')
    plt.legend(loc="lower right")
    plt.savefig('CNN_With_Feature_Selection_ROC.png')  # Save ROC curve
    #plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.title('CNN With Feature Selection Confusion Matrix')
    plt.savefig('CNN_With_Feature_Selection_Confusion_Matrix.png')  # Save confusion matrix
    #plt.show()

def plot_training_accuracy(history, title='Model Accuracy', show_grid=True):
    """
    Plots the training and validation accuracy from a Keras model training history.
    
    Parameters:
    - history: Return value from model.fit() method, which is a Keras History object.
    - title: str, optional. Title of the graph.
    - show_grid: bool, optional. If True, displays a grid on the graph.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation')
    
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    
    if show_grid:
        plt.grid(True)
    plt.savefig('CNN_With_Feature_Selection_Accuracy.png')
    #plt.show()

# Main script using argparse to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CNN model on sensor data.')
    parser.add_argument('data_path', type=str, help='Path to the dataset directory.')
    args = parser.parse_args()

    base_path = args.data_path
    data, labels, scaler, le = load_and_preprocess_data(base_path)

    if data is not None and labels is not None:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        num_classes = len(np.unique(labels))

        model = build_cnn_model(X_train.shape[1:], num_classes)
        history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=64)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        

        y_pred = model.predict(X_test)
        y_test_binarized = label_binarize(y_test, classes=[i for i in range(num_classes)])
        class_names = [label for label in label_mapping.values()]

        plot_roc_curve(y_test_binarized, y_pred, num_classes, class_names)
        plot_confusion_matrix(y_test, y_pred.argmax(axis=1), class_names)
        plot_training_accuracy(history, title='Training and Validation Accuracy', show_grid=True)
    else:
        print("Failed to load data.")