import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from LSTM_updated import MultitaskLSTM

# 1. Load test CSV
df = pd.read_csv("testing_5_edited.csv", header=None)

# 2. Extract string labels
y_true = df.iloc[:, 0].astype(str).str.strip().values

# 3. Extract complex I/Q values
complex_data = df.iloc[:, 1:].applymap(lambda x: complex(x.replace(' ', '').replace('+-', '-'))).values
I = np.real(complex_data)
Q = np.imag(complex_data)

# 4. Stack into [N, 128, 2]
X_np = np.stack([I, Q], axis=-1)
X = torch.tensor(X_np, dtype=torch.float32)

# 5. Normalize by max amplitude
X = X / 876.5683  # manually computed max amplitude

# 6. Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultitaskLSTM().to(device)
model.load_state_dict(torch.load("best_lstm_weights.pt", map_location=device))
model.eval()

# 7. Predict
with torch.no_grad():
    preds_mod, _ = model(X.to(device))
    pred_indices = preds_mod.argmax(dim=1).cpu().numpy()

# 8. Map indices back to label strings
unique_labels = np.unique(y_true)
label_to_index = {label: i for i, label in enumerate(unique_labels)}
index_to_label = {i: label for label, i in label_to_index.items()}
y_pred = [index_to_label[idx] for idx in pred_indices]

# 9. Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")

# 10. Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=unique_labels)
plt.title("Modulation Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
