import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from itertools import cycle

# Define label mapping
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

def calculate_total_acceleration(df):
    """Calculate total acceleration vector magnitude."""
    return np.sqrt(df['Acc_X']**2 + df['Acc_Y']**2 + df['Acc_Z']**2)

def extract_features(segment):
    """Extract statistical features from a segment of sensor data."""
    N = len(segment)
    mean = np.mean(segment)
    variance = np.var(segment)
    skewness = np.sum((segment - mean)**3) / (N * np.std(segment)**3)
    kurtosis = np.sum((segment - mean)**4) / (N * np.std(segment)**4)
    
    features = {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'min': np.min(segment),
        'max': np.max(segment)
    }
    
    autocorrs = np.correlate(segment - mean, segment - mean, mode='full')[N-1:N+10] / (N - np.arange(0, 11))
    features.update({f'autocorr_{i}': autocorr for i, autocorr in enumerate(autocorrs)})
    
    dft = np.fft.fft(segment)
    spectrum = np.abs(dft)
    frequencies = np.fft.fftfreq(N, d=1/25)
    peaks, _ = find_peaks(spectrum)
    top_peaks = peaks[:5] if len(peaks) >= 5 else peaks
    
    features.update({f'peak_{i}': spectrum[peak] for i, peak in enumerate(top_peaks)})
    features.update({f'freq_{i}': frequencies[peak] for i, peak in enumerate(top_peaks)})
    
    return features

def process_sensor_data(df, peak_index):
    """Extracts a window of data around the peak index for all axes."""
    window_size = 50  # 50 samples before and after the peak
    start_index = max(0, peak_index - window_size)
    end_index = min(peak_index + window_size + 1, len(df))
    return df.iloc[start_index:end_index]

def load_and_preprocess_data(base_path):
    """Load data from multiple sensors and preprocess it."""
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
                    df = pd.read_csv(sensor_file_path)
                    
                    total_acc = calculate_total_acceleration(df[['Acc_X', 'Acc_Y', 'Acc_Z']])
                    reference_peak_index = np.argmax(total_acc)
                    windowed_data = process_sensor_data(df, reference_peak_index)

                    for column in ['Acc_X', 'Acc_Y', 'Acc_Z']:
                        features = extract_features(windowed_data[column])
                        combined_features.extend(list(features.values()))

                if combined_features:
                    expected_length = 100  # Adjust as needed
                    if len(combined_features) != expected_length:
                        combined_features = combined_features[:expected_length] + [0] * (expected_length - len(combined_features))

                    data.append(combined_features)
                    labels.append(label)

    if not data:
        print("No data could be collected. Please check the dataset and paths.")
        return None

    data = np.array(data, dtype=float)
    data = np.nan_to_num(data)  # Replace NaNs and Inf values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    return data, labels, scaler, le

#with padding step
# def load_and_preprocess_data(base_path):
#     """Load data from multiple sensors and preprocess it."""
#     data = []
#     labels = []
#     window_size = 101  # Ensure the window size is consistent

#     for label_folder in os.listdir(base_path):
#         label_id = label_folder.split('-')[0]
#         label = label_mapping.get(label_id)
#         if label is None:
#             continue
#         label_folder_path = os.path.join(base_path, label_folder)

#         for volunteer_folder in os.listdir(label_folder_path):
#             volunteer_folder_path = os.path.join(label_folder_path, volunteer_folder)

#             for test_folder in os.listdir(volunteer_folder_path):
#                 test_folder_path = os.path.join(volunteer_folder_path, test_folder)
#                 combined_features = []

#                 for sensor_file in os.listdir(test_folder_path):
#                     sensor_file_path = os.path.join(test_folder_path, sensor_file)
#                     df = pd.read_csv(sensor_file_path)
#                     df = df.fillna(0)  # Fill NaN values with 0
#                     df = df.replace([np.inf, -np.inf], 0)  # Replace infinite values with 0
                    
#                     total_acc = calculate_total_acceleration(df[['Acc_X', 'Acc_Y', 'Acc_Z']])
#                     reference_peak_index = np.argmax(total_acc)
#                     windowed_data = process_sensor_data(df, reference_peak_index)
                    
#                     if len(windowed_data) < window_size:
#                         # Pad the windowed data with zeros if it's shorter than the window size
#                         padding = pd.DataFrame(np.zeros((window_size - len(windowed_data), windowed_data.shape[1])), columns=windowed_data.columns)
#                         windowed_data = pd.concat([windowed_data, padding], axis=0)

#                     for column in ['Acc_X', 'Acc_Y', 'Acc_Z']:
#                         features = extract_features(windowed_data[column])
#                         combined_features.extend(list(features.values()))

#                 if combined_features:
#                     expected_length = 100  # Adjust as needed
#                     if len(combined_features) != expected_length:
#                         combined_features = combined_features[:expected_length] + [0] * (expected_length - len(combined_features))

#                     data.append(combined_features)
#                     labels.append(label)

    # if not data:
    #     print("No data could be collected. Please check the dataset and paths.")
    #     return None, None, None, None

    # data = np.array(data, dtype=float)
    # data = np.nan_to_num(data)  # Replace NaNs and Inf values

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data = scaler.fit_transform(data)

    # le = LabelEncoder()
    # labels = le.fit_transform(labels)

    # return data, labels, scaler, le

def build_lstm_model(input_shape, num_classes, num_samples):
    Ni = input_shape[1]  # Number of features per timestep
    No = num_classes     # Number of output classes
    Ns = num_samples     # Number of samples in the training dataset
    alpha = 5            # Adjust based on model complexity and generalization needs
    
    Nh = int(Ns / (alpha * (Ni + No)))  # Calculate optimized number of LSTM nodes

    print(f"Optimized number of LSTM nodes: {Nh}")

    model = Sequential([
        LSTM(Nh, return_sequences=True, input_shape=input_shape),  # Use calculated Nh
        Dropout(0.2),
        LSTM(Nh),  # Second LSTM layer with Nh nodes, not returning sequences
        Dropout(0.2),
        Dense(100, activation='relu'),  # An additional dense layer
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
    plt.title('LSTM_V2 ROC')
    plt.legend(loc="lower right")
    plt.savefig('LSTM_With_Feature_Selection_ROC.png')  # Save ROC curve
    #plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.yticks(rotation=45)
    plt.grid(False)
    plt.title('LSTM_V2 Confusion Matrix')
    plt.savefig('LSTM_With_Feature_Selection_Confusion_Matrix.png')  # Save confusion matrix
    plt.show()
    plt.close()

def plot_training_accuracy(history, title='LSTM_V2 Model Accuracy', show_grid=True):
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
    plt.savefig('LSTM_With_Feature_Selection_Accuracy.png')
    #plt.show()
    plt.close()

# Main script using argparse to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an LSTM model on sensor data.')
    parser.add_argument('data_path', type=str, help='Path to the dataset directory.')
    args = parser.parse_args()

    base_path = args.data_path
    data, labels, scaler, le = load_and_preprocess_data(base_path)

    if data is not None and labels is not None:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        num_classes = len(np.unique(labels))
        num_samples = X_train.size
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), num_classes,num_samples)
        history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=64)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        final_train_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        print(f"Final Training Accuracy: {final_train_accuracy*100:.2f}%")
        print(f"Final Validation Accuracy: {final_val_accuracy*100:.2f}%")

        y_pred = model.predict(X_test)
        y_test_binarized = label_binarize(y_test, classes=[i for i in range(num_classes)])
        class_names = [label for label in label_mapping.values()]

        plot_roc_curve(y_test_binarized, y_pred, num_classes, class_names)
        plot_confusion_matrix(y_test, y_pred.argmax(axis=1), class_names)
        plot_training_accuracy(history, show_grid=True)
    else:
        print("Failed to load data.")
