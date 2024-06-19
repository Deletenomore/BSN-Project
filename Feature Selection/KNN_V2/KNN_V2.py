#Insturction to run this code
#1. Create a virtual environment: venv: python -m venv myenv (replaced myenv as needed ) | conda: conda create -n myenv python= "version number"
#2. Activate the virtual environment, windows: myenv\Scripts\activate | macOS and Linux: source myenv/bin/activate
#3. Initialize the environment by requirements.txt: pip install -r requirements.txt
#4. Run the code: python KNN_V2.py "datapath"

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import argparse
from itertools import cycle

# Define label mapping globally
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
    features = {
        'min': np.min(segment),
        'max': np.max(segment),
        'mean': mu,
        'variance': np.var(segment, ddof=0),
        'skewness': skew(segment),
        'kurtosis': kurtosis(segment)
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
                    for column in sensor_data.columns:
                        if column != 'Time':
                            features = calculate_features(sensor_data[column])
                            combined_features.extend(list(features.values()))
                
                if combined_features:
                    expected_length = 100
                    if len(combined_features) != expected_length:
                        combined_features = combined_features[:expected_length] + [0] * (expected_length - len(combined_features))
                    data.append(combined_features)
                    labels.append(label)
                else:
                    print(f"No features extracted for {test_folder_path}")

    if not data:
        print("No data could be collected, please check the dataset and paths.")
        return None, None, None, None

    data = np.array(data, dtype=float)
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data)

    labels = np.array(labels)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return data, labels, scaler, le

def build_knn_model(X_train, y_train):
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def plot_roc_curve(y_test, y_score, num_classes, labels):
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('KNN With Feature Selection Classifier ROC')
    plt.legend(loc="lower right")
    plt.savefig('KNN_With_Feature_Selection_ROC.png')  # Save ROC curve
    #plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.title('KNN With Feature Selection Confusion Matrix')
    plt.savefig('KNN_With_Feature_Selection_Confusion_Matrix.png')  # Save confusion matrix
    #plt.show()
    plt.close()

def print_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validated score:", grid_search.best_score_)
    return grid_search.best_estimator_

def plot_validation_curve(X_train, y_train):
    param_range = np.arange(1, 16)
    train_scores, test_scores = validation_curve(
        KNeighborsClassifier(), X_train, y_train, 
        param_name="n_neighbors", param_range=param_range, 
        cv=5, scoring="accuracy", n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    plt.title("Validation Curve with KNN")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("Validation Curve with KNN.png")
    #plt.show()
    plt.close()

def main(data_path):
    data, labels, scaler, le = load_and_preprocess_data(data_path)
    if data is None or labels is None:
        print("Data loading or preprocessing failed.")
        return

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    knn_model = build_knn_model(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    y_scores = knn_model.predict_proba(X_test)
    class_names = [label for label in label_mapping.values()]
    y_test_binarized = label_binarize(y_test, classes=np.unique(labels))

    plot_roc_curve(y_test_binarized, y_scores, len(np.unique(labels)), class_names)
    plot_confusion_matrix(y_test, y_pred, class_names)
    #print_accuracy(y_test, y_pred)

    #find the best N neighbors
    best_knn = tune_hyperparameters(X_train, y_train)
    plot_validation_curve(X_train, y_train)
    
    # Testing the best model
    y_pred = best_knn.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a KNN model on sensor data.')
    parser.add_argument('data_path', type=str, help='Path to the dataset directory.')
    args = parser.parse_args()
    main(args.data_path)
