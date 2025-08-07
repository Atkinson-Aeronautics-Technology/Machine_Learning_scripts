# Required Libraries
import numpy as np  # For numerical operations on arrays
import torch  # Main PyTorch library for tensor computations and GPU acceleration
import torch.nn as nn  # Neural network layers
import torch.optim as optim  # Optimization algorithms like Adam
from torch.utils.data import DataLoader, TensorDataset, random_split  # Tools for managing datasets
from sklearn.metrics import ConfusionMatrixDisplay  # For plotting confusion matrix
import matplotlib.pyplot as plt  # Plotting library for evaluation

# --------------------------------------
# 1. Define the Multi-Task LSTM Model
# --------------------------------------
class MultitaskLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2,
                 num_mod_classes=6, num_sig_classes=8, dropout=0.3):
        """
        This class defines the neural network model.
        It uses an LSTM layer to process signal data and then predicts both modulation and signal type.

        Parameters:
        - input_size: Number of input features per time step (2 → I and Q channels)
        - hidden_size: Number of LSTM units
        - num_layers: Number of stacked LSTM layers
        - num_mod_classes: Number of modulation types (output dimension for task 1)
        - num_sig_classes: Number of signal types (output dimension for task 2)
        - dropout: Probability of dropout for regularization
        """
        super().__init__()

        # Bidirectional LSTM reads the signal both forward and backward
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        # Dropout layer helps prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layers for each task
        self.fc_mod = nn.Linear(hidden_size * 2, num_mod_classes)  # Task 1: modulation classification
        self.fc_sig = nn.Linear(hidden_size * 2, num_sig_classes)  # Task 2: signal type classification

    def forward(self, x):
        """
        Forward pass of the model.
        Input shape: [batch_size, time_steps=128, features=2]

        Returns:
        - logits for modulation classes
        - logits for signal type classes
        """
        lstm_out, _ = self.lstm(x)  # Output shape: [batch_size, 128, 256]
        pooled = torch.mean(lstm_out, dim=1)  # Mean pooling across time dimension → [batch_size, 256]
        pooled = self.dropout(pooled)  # Apply dropout
        return self.fc_mod(pooled), self.fc_sig(pooled)  # Return both predictions
# -------------------------------------------------------
# 2. Training Code — Runs only when script is executed
# -------------------------------------------------------
if __name__ == "__main__":

    # Load the dataset (preprocessed I/Q signal samples and labels)
    data = np.load("radcom_preprocessed.npz")
    X = torch.tensor(data["X"], dtype=torch.float32)  # Shape: [samples, 128, 2] (time × I/Q)
    y_mod = torch.tensor(data["y_mod"], dtype=torch.long)  # Labels for modulation
    y_sig = torch.tensor(data["y_sig"], dtype=torch.long)  # Labels for signal type

    # Normalize the input data to improve training performance
    X_mean = X.mean(dim=(0, 1), keepdim=True)
    X_std = X.std(dim=(0, 1), keepdim=True)
    X = (X - X_mean) / (X_std + 1e-6)  # Standard normalization

    # Save normalization statistics so that the test data can use the same scaling
    torch.save({'mean': X_mean, 'std': X_std}, "norm_stats.pt")
    print("Normalization stats saved to norm_stats.pt")

    # Sanity check: ensure input shape is [N, 128, 2]
    if X.shape[1] != 128 or X.shape[2] != 2:
        X = X.permute(0, 2, 1)  # Reorder axes if necessary

    # Ensure that labels are within valid class range
    assert y_mod.min() >= 0 and y_mod.max() < 6
    assert y_sig.min() >= 0 and y_sig.max() < 8
    # Split dataset into training (80%) and testing (20%) sets
    dataset = TensorDataset(X, y_mod, y_sig)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Create DataLoaders to handle batch processing during training
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128)
    # Set up the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model = MultitaskLSTM().to(device)  # Move model to device
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # LR decay every 5 epochs
    # -------------------------------
    # 3. Training Loop Begins
    # -------------------------------
    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb_mod, yb_sig in train_loader:
            xb, yb_mod, yb_sig = xb.to(device), yb_mod.to(device), yb_sig.to(device)

            # Forward pass → two predictions: modulation and signal type
            pred_mod, pred_sig = model(xb)

            # Compute total loss as the sum of the two tasks
            loss = criterion(pred_mod, yb_mod) + criterion(pred_sig, yb_sig)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        # Print average loss for this epoch
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.6f}")

        # Step the learning rate scheduler
        scheduler.step()
    # -------------------------------
    # 4. Evaluation on Test Set
    # -------------------------------
    model.eval()
    all_true_mod, all_pred_mod = [], []
    all_true_sig, all_pred_sig = [], []

    with torch.no_grad():  # Disable gradients for evaluation
        for xb, yb_mod, yb_sig in test_loader:
            xb = xb.to(device)
            pred_mod, pred_sig = model(xb)

            # Save ground truth and predicted labels
            all_true_mod.extend(yb_mod.numpy())
            all_pred_mod.extend(pred_mod.argmax(dim=1).cpu().numpy())
            all_true_sig.extend(yb_sig.numpy())
            all_pred_sig.extend(pred_sig.argmax(dim=1).cpu().numpy())
    # -------------------------------
    # 5. Plotting Confusion Matrices
    # -------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Modulation confusion matrix
    ConfusionMatrixDisplay.from_predictions(all_true_mod, all_pred_mod, ax=axs[0])
    axs[0].set_title("Modulation Classification")

    # Signal type confusion matrix
    ConfusionMatrixDisplay.from_predictions(all_true_sig, all_pred_sig, ax=axs[1])
    axs[1].set_title("Signal Type Classification")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png")  # Save the figure for documentation
    plt.close()
    # -------------------------------
    # 6. Save Trained Model Weights
    # -------------------------------
    torch.save(model.state_dict(), "best_lstm_weights.pt")
    print("Model weights saved to best_lstm_weights.pt")
