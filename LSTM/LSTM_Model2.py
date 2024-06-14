import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import glob

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_data_from_directory(directory_path):
    file_paths = glob.glob(os.path.join(directory_path, '**/*.csv'), recursive=True)
    return file_paths

def preprocess_data(file_path, window_size, step_size):
    data = pd.read_csv(file_path)
    
    # Ensure no NaN or infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    
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

def plot_confusion_matrix(true_classes, predicted_classes, labels):
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:")
    print(cm)
    
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the confusion matrix plot as an image file
    plt.close()


def plot_roc_curve(true_classes, predictions, labels):
    y_test_binarized = label_binarize(true_classes, classes=np.arange(len(labels)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure()
    for i in range(len(labels)):
        if np.sum(y_test_binarized[:, i]) == 0:
            print(f"No positive samples in class {i}, skipping ROC curve for this class.")
            continue
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='ROC curve (class %d, area = %0.2f)' % (i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')  # Save the ROC curve plot as an image file
    plt.close()

def plot_training_accuracy(train_accuracies, val_accuracies):
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig('training_accuracy.png')  # Save the training accuracy plot as an image file
    plt.close()

def train_and_evaluate_model(data_by_label, input_size, hidden_size, output_size, num_layers=2, dropout=0.5, learning_rate=0.001, batch_size=64, num_epochs=20):
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

    dataset = TimeSeriesDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

    plot_confusion_matrix(all_labels, all_preds,labels)

    # Convert all_preds to one-hot encoding for ROC curve plotting
    all_preds_one_hot = np.eye(len(labels))[all_preds]
    plot_roc_curve(all_labels, all_preds_one_hot, labels)
    plot_training_accuracy(train_accuracies, val_accuracies)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <directory_path>")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    directory_path = sys.argv[1]
    window_size = 25
    step_size = 12
    input_size = 9
    hidden_size = 100  # Increased hidden size for better learning capacity
    output_size = 10
    num_layers = 3  # Increased the number of layers
    dropout = 0.3  # Reduced dropout for retaining more information
    learning_rate = 0.001  # Reduced learning rate for finer updates
    batch_size = 32  # Reduced batch size for more frequent updates
    num_epochs = 10  # Increased number of epochs

    file_paths = load_data_from_directory(directory_path)
    data_by_label = organize_data(file_paths, window_size, step_size)
    train_and_evaluate_model(data_by_label, input_size, hidden_size, output_size, num_layers, dropout, learning_rate, batch_size, num_epochs)
