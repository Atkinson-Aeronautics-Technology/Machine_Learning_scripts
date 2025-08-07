# ---------------------------------------------------------------
# ✅ REQUIRED LIBRARIES
# ---------------------------------------------------------------
import torch                         # PyTorch library for running trained neural networks
import pandas as pd                  # For loading and parsing CSV files
import numpy as np                   # Numerical operations on arrays
import matplotlib.pyplot as plt      # To display confusion matrix
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay  # Evaluation metrics
from LSTM_updated import MultitaskLSTM  # This imports the trained multitask LSTM model architecture
# ---------------------------------------------------------------
# ✅ STEP 1: LOAD TEST CSV
# ---------------------------------------------------------------
# The CSV should be formatted as:
# - Column 0: ground truth labels (modulation names like "AM-DSB", "FM", etc.)
# - Columns 1–128: signal samples formatted as complex numbers (I + jQ)
df = pd.read_csv("testing_5_edited.csv", header=None)
# ---------------------------------------------------------------
# ✅ STEP 2: EXTRACT GROUND TRUTH LABELS
# ---------------------------------------------------------------
# Convert first column (modulation labels) to string and strip whitespace
# These will be used to evaluate prediction accuracy later
y_true = df.iloc[:, 0].astype(str).str.strip().values
# ---------------------------------------------------------------
# ✅ STEP 3: PARSE COMPLEX SIGNAL DATA
# ---------------------------------------------------------------
# Columns 1 to end are signal samples as complex numbers in string form
# We convert them to actual Python complex numbers and then separate
# them into real (I) and imaginary (Q) parts.

complex_data = df.iloc[:, 1:].applymap(
    lambda x: complex(x.replace(' ', '').replace('+-', '-'))  # sanitize string to parse correctly
).values

I = np.real(complex_data)  # In-phase component
Q = np.imag(complex_data)  # Quadrature component
# ---------------------------------------------------------------
# ✅ STEP 4: STACK I AND Q INTO [N, 128, 2] SHAPE
# ---------------------------------------------------------------
# We stack the I and Q components along the last dimension, so each signal
# sample becomes a sequence of 128 time steps with 2 channels (I and Q).
# This is the required input format for the LSTM model.
X_np = np.stack([I, Q], axis=-1)

# Convert to PyTorch tensor for model input
X = torch.tensor(X_np, dtype=torch.float32)
# ---------------------------------------------------------------
# ✅ STEP 5: NORMALIZE SIGNAL AMPLITUDES
# ---------------------------------------------------------------
# During training, we observed that SCEPTRE signals had much larger amplitude than RadCom.
# To standardize input scale, we divide by the maximum amplitude observed in any training signal.
# I tried using the Z normalization method like in the LSTM file but it did not work. The maximum amplitude one doesn't work either. Good luck finding some other technique!
X = X / 876.5683
# ---------------------------------------------------------------
# ✅ STEP 6: LOAD TRAINED LSTM MODEL AND WEIGHTS
# ---------------------------------------------------------------
# We move the model to GPU if available, then load previously trained weights.
# Note: The model architecture *must* match the one used during training.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultitaskLSTM().to(device)

# Load the saved weights into the model
model.load_state_dict(torch.load("best_lstm_weights.pt", map_location=device))

# Set model to evaluation mode — disables dropout, etc.
model.eval()
# ---------------------------------------------------------------
# ✅ STEP 7: PERFORM PREDICTIONS ON TEST SIGNALS
# ---------------------------------------------------------------
# We only use the modulation head here (we're not evaluating signal type)
# torch.no_grad() prevents PyTorch from tracking gradients (saves memory)
with torch.no_grad():
    preds_mod, _ = model(X.to(device))  # Discard signal type predictions
    pred_indices = preds_mod.argmax(dim=1).cpu().numpy()  # Pick the class with highest probability
# ---------------------------------------------------------------
# ✅ STEP 8: CONVERT PREDICTED INDICES BACK TO TEXT LABELS
# ---------------------------------------------------------------
# Since y_true is a string (e.g., "AM-DSB"), we map predicted index values
# back to strings using the same label order that appears in y_true.
# ⚠️ Assumes all label types present in y_true are seen during training.

unique_labels = np.unique(y_true)  # All label types in ground truth
label_to_index = {label: i for i, label in enumerate(unique_labels)}  # str → int
index_to_label = {i: label for label, i in label_to_index.items()}    # int → str

# Map each predicted index back to corresponding label string
y_pred = [index_to_label[idx] for idx in pred_indices]
# ---------------------------------------------------------------
# ✅ STEP 9: CALCULATE AND PRINT ACCURACY
# ---------------------------------------------------------------
# Simple classification accuracy = correct predictions / total predictions
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
# ---------------------------------------------------------------
# ✅ STEP 10: PLOT CONFUSION MATRIX
# ---------------------------------------------------------------
# This gives a visual summary of how well the model is performing across each class.
# Diagonal = correct predictions; Off-diagonal = confusion between classes
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=unique_labels)
plt.title("Modulation Confusion Matrix")
plt.xticks(rotation=45)  # Rotate labels to avoid overlapping
plt.tight_layout()
plt.show()
