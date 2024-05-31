import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import joblib
import argparse
import seaborn as sns

# Define the label mapping
label_mapping = {
    '901': 'front-sitting',
    '902': 'front-protecting',
    '903': 'front-knees',
    '904': 'front-knees-lying',
    '905': 'front-quick-recovery',
    '906': 'front-slow-recovery',
    '907': 'front-right',
    '908': 'front-left',
    '909': 'back-sitting',
    '910': 'back-knees'
}

# Function to load data from the dataset structure
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

# Function to preprocess data
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
    return X_train, X_test, y_train, y_test, scaler, le, le.classes_

# Function to train Decision Tree model and evaluate with ROC AUC
def train_and_evaluate(X_train, X_test, y_train, y_test, classes,le):
    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X_train, y_train)

    # Predict class labels, not probabilities
    y_pred = dtree.predict(X_test) 

    # Binarize the labels for the entire range of actual class labels
    y_test_binarized = label_binarize(y_test, classes=classes)

    # Predict probabilities
    y_pred_proba = dtree.predict_proba(X_test)
    
    # Plot ROC Curve
    plot_multi_class_roc(y_test_binarized, y_pred_proba, classes,le)

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test,  y_pred , [le.inverse_transform([i])[0] for i in range(len(classes))])

    return dtree

def plot_multi_class_roc(y_test, y_pred_proba, classes, le):
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, label in enumerate(classes):
        fpr[label], tpr[label], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])

    # Plot all ROC curves
    plt.figure()
    for i in range(len(classes)):
        label = le.inverse_transform([classes[i]])[0]  # Use inverse_transform to get original labels
        plt.plot(fpr[classes[i]], tpr[classes[i]], label=f'ROC curve of class {label} (area = {roc_auc[classes[i]]:.2f})')

    plt.title('Multi-class ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("Decision Tree ROC.png")
    #plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Highlighting each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.5, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig("Decision Tree Confusion_matrix.png")
    plt.close()  # Close the plot to free up memory

# Function to save model and scaler
def save_model(dtree_model, scaler, le, model_path='dtree_model.joblib', scaler_path='scaler.joblib', le_path='label_encoder.joblib'):
     # Create directory if not exists
    base_dir = "trained_model"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Define full paths for saving model components
    full_model_path = os.path.join(base_dir, model_path)
    full_scaler_path = os.path.join(base_dir, scaler_path)
    full_le_path = os.path.join(base_dir, le_path)

    # Save the model, scaler, and label encoder in the specified directory
    joblib.dump(dtree_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)

def preprocess_new_data(new_data_path, scaler):
    try:
        # Load new data and store each sensor data's flattened array
        new_data = []
        for sensor_file in os.listdir(new_data_path):
            sensor_file_path = os.path.join(new_data_path, sensor_file)
            sensor_data = pd.read_csv(sensor_file_path)
            # Flatten the data and normalize the length to the median length of training data or another consistent measure
            flattened_data = sensor_data.values.flatten()
            new_data.append(flattened_data)
        
        # Concatenate all sensor data into one array
        new_data = np.concatenate(new_data)
        
        # Trim or expand the data to match the expected number of features
        required_features = scaler.n_features_in_
        current_features = len(new_data)
        
        if current_features < required_features:
            # If less, consider truncating or finding an appropriate method to handle missing data
            new_data = np.pad(new_data, (0, required_features - current_features), 'constant')
        elif current_features > required_features:
            # If more, truncate the data
            new_data = new_data[:required_features]
        
        # Reshape new_data for a single sample
        new_data = new_data.reshape(1, -1)
        
        # Scale the data using the loaded scaler
        new_data_scaled = scaler.transform(new_data)

        return new_data_scaled
    except Exception as e:
        print(f"An error occurred during preprocessing: {str(e)}")
        raise


def predict_new_data(new_data_path, model_path, scaler_path, le_path):
    try:
        dtree_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)

        new_data_scaled = preprocess_new_data(new_data_path, scaler)
        prediction = dtree_model.predict(new_data_scaled)
        predicted_label = le.inverse_transform(prediction)

        return predicted_label
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Predict using Decision Tree model for fall detection")
    parser.add_argument('mode', choices=['train', 'predict'], help="Mode: train a new model or predict using an existing model")
    parser.add_argument('--dataset', type=str, help="Path to the dataset directory for training")
    parser.add_argument('--new_data', type=str, help="Path to the new data directory for prediction")
    parser.add_argument('--model', type=str, default='trained_model/dtree_model.joblib', help="Path to save or load the model")
    parser.add_argument('--scaler', type=str, default='trained_model/scaler.joblib', help="Path to save or load the scaler")
    parser.add_argument('--label_encoder', type=str, default='trained_model/label_encoder.joblib', help="Path to save or load the label encoder")

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.dataset:
            print("Please provide the path to the dataset using --dataset")
            exit(1)
        X_train, X_test, y_train, y_test, scaler, le, classes = preprocess_data(args.dataset)
        
        

        classes = np.unique(y_train)  # Get the list of unique classes
        dtree_model = train_and_evaluate(X_train, X_test, y_train, y_test, classes,le)
        
        # Evaluate the model on the test set
        y_test_pred = dtree_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f'Test Accuracy: {test_accuracy:.2f}')
        
        save_model(dtree_model, scaler, le, args.model, args.scaler, args.label_encoder)
        
    elif args.mode == 'predict':
        if not args.new_data:
            print("Please provide the path to the new data using --new_data")
            exit(1)
        prediction = predict_new_data(args.new_data, args.model, args.scaler, args.label_encoder)
        print(f'Predicted Fall Action Label: {prediction[0]}')
