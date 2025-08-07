# --------------------------------------------------------
# ✅ IMPORTS: Required Libraries
# --------------------------------------------------------
import torch                      # Core PyTorch functionality
import torch.nn as nn             # Neural network layers
from torch.utils.data import Dataset, DataLoader  # For creating and loading custom datasets
import pandas as pd               # CSV/Excel I/O
import os                        # File and directory handling
import matplotlib.pyplot as plt   # Plotting metrics if needed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Performance evaluation
import numpy as np
import torch.nn.functional as F   # Useful functional tools like ReLU
# --------------------------------------------------------
# ✅ SET FIXED LABELS: Control what classes are used
# --------------------------------------------------------
# These labels represent which class IDs are valid.
# They MUST match the dataset's structure and model expectations.
Fixed_labels = [
    0, 1, 2, 3, 4,
    10, 11, 12, 13, 14,
    20, 21, 22, 23, 24,
    30, 31, 32, 34,
    40, 41, 44,
    50, 51, 54,
    61
]

# Create a dictionary that maps raw label (e.g., 44) → index (e.g., 21)
LABEL_MAP = {label: idx for idx, label in enumerate(Fixed_labels)}

# The number of output classes for classification
NUM_CLASSES = len(Fixed_labels)
# --------------------------------------------------------
# ✅ CUSTOM DATASET CLASS: Loads signal slices from CSV
# --------------------------------------------------------
# This class is responsible for:
# - Reading in signal data (as I/Q complex values)
# - Reading corresponding labels and SNR values
# - Filtering out invalid labels
# - Converting signals to [2, 128] tensors for CNN input
class SignalSetDataset(Dataset):
    def __init__(self, data_path, labels_path, snr_path, first_row=0, last_row=9999):
        assert last_row >= first_row, "last row must be greater than first row"
        
        # Calculate how many rows to read and which ones to skip
        nrows = last_row - first_row + 1
        skiprows = list(range(1, first_row + 1))  # skip everything before `first_row`
        
        # Load data from CSVs
        self.signals_df = pd.read_csv(data_path, header=None, skiprows=skiprows, nrows=nrows)
        self.labels_df = pd.read_csv(labels_path, header=None, skiprows=skiprows, nrows=nrows)
        self.snr_df = pd.read_csv(snr_path, header=None, skiprows=skiprows, nrows=nrows)
        
        # Only keep rows that have a valid label
        original_labels = self.labels_df[0]
        mask = original_labels.isin(Fixed_labels)
        self.signals_df = self.signals_df[mask].reset_index(drop=True)
        original_labels = original_labels[mask].reset_index(drop=True)
        self.snr_df = self.snr_df[mask].reset_index(drop=True)

        # Parse I/Q complex numbers from strings like "0.3+0.2i"
        parsed_signals = self.signals_df.apply(
            lambda row: [complex(val.replace("i", "j")) for val in row],
            axis=1
        )
        real_parts = [torch.tensor([c.real for c in row], dtype=torch.float32) for row in parsed_signals]
        imag_parts = [torch.tensor([c.imag for c in row], dtype=torch.float32) for row in parsed_signals]
        
        # Final input tensor: [N, 2, 128] → CNN expects channel-first
        self.X = torch.stack([
            torch.stack([real, imag], dim=0)
            for real, imag in zip(real_parts, imag_parts)
        ])

        # Convert raw labels to 0-indexed class numbers
        mapped_labels = original_labels.map(LABEL_MAP)
        self.labels = torch.tensor(mapped_labels.values, dtype=torch.long)

        # SNR (unused here, but loaded for potential future use)
        self.snr = torch.tensor(self.snr_df.values.flatten(), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]
# --------------------------------------------------------
# ✅ CNN MODEL: 5-layer 1D convolutional architecture
# --------------------------------------------------------
# The model processes each signal as 2 input channels (I and Q),
# applies multiple convolution + pooling layers to extract features,
# then uses fully connected layers to output classification scores.

class SignalClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(SignalClassifierCNN, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        # Conv Block 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        # Conv Block 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        # Conv Block 4
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)

        # Final Conv Block
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=2)
        self.bn5 = nn.BatchNorm1d(256)

        # Adaptive average pooling brings output to shape [batch_size, 256, 1]
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # Fully connected classifier
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)  # Final prediction logits
# --------------------------------------------------------
# ✅ TRAINING LOOP SETUP
# --------------------------------------------------------

directory = r"C:\Users\ShaileeKulkarni\Downloads\CNN stuff"  # Root folder containing CSVs
total_rows = 520000       # Total rows in dataset
chunk_size = 10000        # Number of rows to process per training loop
num_epochs_per_chunk = 25  # Epochs per chunk
model_save_path = os.path.join(directory, "cnn_weights.pth")  # Where weights will be saved
history = []  # To store loss and accuracy over time
# --------------------------------------------------------
# ✅ Initialize model, loss function, optimizer, scheduler
# --------------------------------------------------------
model = SignalClassifierCNN(num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs

# Load pre-trained weights if they exist
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print("Previous weights have been loaded")
# --------------------------------------------------------
# ✅ MAIN TRAINING LOOP (chunk-by-chunk)
# --------------------------------------------------------
for chunk_start in range(0, total_rows, chunk_size):
    chunk_end = min(chunk_start + chunk_size - 1, total_rows - 1)
    print(f"\n Training on {chunk_start} to {chunk_end}")

    # Load chunk into custom dataset
    dataset = SignalSetDataset(
        data_path=os.path.join(directory, "train_data.csv"),
        labels_path=os.path.join(directory, "train_labels.csv"),
        snr_path=os.path.join(directory, "train_snr.csv"),
        first_row=chunk_start,
        last_row=chunk_end
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train on this chunk for N epochs
    for epoch in range(num_epochs_per_chunk):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        avg_loss = running_loss / len(loader)
        accuracy = 100.0 * correct / total
        print(f"Epoch: [{epoch + 1}/{num_epochs_per_chunk}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Save history for analysis
        history.append({
            "ChunkStart": chunk_start,
            "ChunkEnd": chunk_end,
            "Epoch": epoch + 1,
            "Loss": avg_loss,
            "Accuracy": accuracy
        })

    # Save model after each chunk
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved model after training rows {chunk_start} to {chunk_end}")
# --------------------------------------------------------
# ✅ EXPORT RESULTS TO EXCEL FOR ANALYSIS
# --------------------------------------------------------
df = pd.DataFrame(history)
df.to_excel(os.path.join(directory, "training_results_fullset.xlsx"), index=False)
