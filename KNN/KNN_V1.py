import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder,label_binarize
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, confusion_matrix,ConfusionMatrixDisplay
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
    return X_train, X_test, y_train, y_test, scaler, le, le.classes_, imputer

def train_and_evaluate(X_train, X_test, y_train, y_test, classes, le):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    #plot_confusion_matrix(y_test, y_pred, [le.inverse_transform([i])[0] for i in range(len(classes))])
    return knn,y_pred

def plot_confusion_matrix(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = le.inverse_transform(range(len(label_mapping)))  # Convert numeric labels back to original string labels
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size for better visibility
    cm_display.plot(cmap=plt.cm.Blues, ax=ax)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.yticks(rotation=45)
    plt.grid(False) 
    plt.title('KNN_V1 Confusion Matrix')
    plt.savefig('KNN_V1_Confusion_Matrix.png')  # Save confusion matrix
    #plt.show()
    plt.close()

def print_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


def save_model(knn_model, scaler, le, imputer, model_path, scaler_path, le_path, imputer_path):
    base_dir = "trained_model/"
    
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Full paths for saving
    full_model_path = os.path.join(base_dir, model_path)
    full_scaler_path = os.path.join(base_dir, scaler_path)
    full_le_path = os.path.join(base_dir, le_path)
    full_imputer_path = os.path.join(base_dir, imputer_path)

    print("Saving model to:", full_model_path)  # Debugging line to check path
    joblib.dump(knn_model, full_model_path)
    joblib.dump(scaler, full_scaler_path)
    joblib.dump(le, full_le_path)
    joblib.dump(imputer, full_imputer_path)

    print("Models and scalers saved successfully to:", base_dir)

def load_model(model_path):
    return joblib.load(model_path)

def preprocess_new_data(new_data_path, scaler, imputer):
    new_data = pd.read_csv(new_data_path)
    new_data = new_data.values.flatten()  # Assuming the data needs to be flattened for prediction
    new_data = new_data.reshape(1, -1)  # Reshape for single sample prediction
    new_data = imputer.transform(new_data)  # Impute missing values if necessary
    new_data_scaled = scaler.transform(new_data)  # Scale data
    return new_data_scaled

def predict(model, new_data_scaled):
    return model.predict(new_data_scaled)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Predict using KNN model for fall detection")
    parser.add_argument('mode', choices=['train', 'predict'], help="Mode: train a new model or predict using an existing model")
    parser.add_argument('--dataset', type=str, help="Path to the dataset directory for training")
    parser.add_argument('--new_data', type=str, help="Path to the new data directory for prediction")
    parser.add_argument('--model', type=str, default='trained_model/knn_model.joblib', help="Path to save or load the model")
    parser.add_argument('--scaler', type=str, default='trained_model/scaler.joblib', help="Path to save or load the scaler")
    parser.add_argument('--label_encoder', type=str, default='trained_model/label_encoder.joblib', help="Path to save or load the label encoder")
    parser.add_argument('--imputer', type=str, default='trained_model/imputer.joblib', help="Path to save or load the imputer")

    args = parser.parse_args()

    if args.mode == 'train':
        X_train, X_test, y_train, y_test, scaler, le, classes, imputer = preprocess_data(args.dataset)
        knn_model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, classes, le)
        plot_confusion_matrix(knn_model,X_test,y_test, le)
        # Now call print_accuracy
        print_accuracy(y_test, y_pred)

        #save_model(knn_model, scaler, le, imputer, args.model, args.scaler, args.label_encoder, args.imputer)
    elif args.mode == 'predict':
        knn_model = load_model(args.model)
        new_data_scaled = preprocess_new_data(args.new_data, scaler, imputer)
        prediction = predict(knn_model, new_data_scaled)
        predicted_label = le.inverse_transform(prediction)
        print(f'Predicted Fall Action Label: {predicted_label[0]}')