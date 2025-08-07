# ------------------------------------------------------
# ✅ STEP 0: Import required libraries
# ------------------------------------------------------
import h5py  # For reading HDF5 files
import numpy as np  # Numerical array processing
from collections import Counter  # For counting label distributions

# ------------------------------------------------------
# ✅ STEP 1: Define mapping from string labels to class indices
# ------------------------------------------------------
# These mappings ensure consistent integer labels for classification.
# ❗ Do NOT change the order unless you retrain the model from scratch.

mod_map = {
    'AM-DSB': 0,
    'AM-SSB': 1,
    'ASK': 2,
    'BPSK': 3,
    'FMCW': 4,
    'PULSED': 5
}

sig_map = {
    'Airborne-detection': 0,
    'Airborne-range': 1,
    'Air-Ground-MTI': 2,
    'Ground mapping': 3,
    'Radar Altimeter': 4,
    'SATCOM': 5,
    'AM radio': 6,
    'short-range': 7
}
# ------------------------------------------------------
# ✅ STEP 2: Set up file path and storage lists
# ------------------------------------------------------
# Ensure path is absolute and correct to avoid file not found errors.
# ❗ If dataset location changes, update `h5_path` accordingly.
h5_path = r"C:\Users\ShaileeKulkarni\Downloads\CNN stuff\LSTM for new data\RadComOta2.45GHz.hdf5"

# Initialize containers for storing usable data
waveforms = []   # will store [128 x 2] I/Q tensors
mod_labels = []  # stores integer-encoded modulation labels
sig_labels = []  # stores integer-encoded signal type labels
snrs = []        # stores SNR values (as integer)
# ------------------------------------------------------
# ✅ STEP 3: Load and filter the HDF5 dataset
# ------------------------------------------------------
# The structure of each key is a Python tuple:
# (modulation_type, signal_type, snr, sample_id)

with h5py.File(h5_path, 'r') as f:
    keys = list(f.keys())  # all dataset keys

    for i, key in enumerate(keys):
        # Decode key if in bytes
        if isinstance(key, bytes):
            key = key.decode("utf-8")

        # Parse the key tuple safely
        try:
            mod, sig, snr_str, sample_str = eval(key)
        except:
            # Skip keys that are malformed or not proper tuples
            continue

        # Ensure only known modulations and signal types are used
        if mod not in mod_map or sig not in sig_map:
            continue

        # Try extracting the waveform; skip corrupted or wrongly sized entries
        try:
            waveform = f[key][:]
            if waveform.shape != (256,):
                continue  # This prevents shape mismatch during stacking
        except:
            continue

        # Split waveform into I and Q channels (128 each)
        I = waveform[:128]
        Q = waveform[128:]
        iq = np.stack([I, Q], axis=1)  # shape: [128, 2]

        # Store processed results
        waveforms.append(iq.astype(np.float32))  # ensures data is float32 (saves memory + consistency)
        mod_labels.append(mod_map[mod])
        sig_labels.append(sig_map[sig])
        snrs.append(int(snr_str))

        # Status print every 10k samples to monitor progress
        if i % 10000 == 0:
            print(f"Processed {i}/{len(keys)}")
# ------------------------------------------------------
# ✅ STEP 4: Save the filtered and cleaned dataset
# ------------------------------------------------------
# All arrays are stacked and saved into a compressed .npz file.
# This file becomes the standardized input to the model training script.
# ❗ If you add more features (e.g., protocol, timestamp), you must:
#     - Add a new list to store those values (e.g., `protocols = []`)
#     - Extend the np.savez() call with that key (e.g., `y_proto=np.array(protocols)`)
#     - Modify the training script to load and use those labels accordingly.

np.savez(
    "radcom_preprocessed.npz",
    X=np.stack(waveforms),         # shape: [N, 128, 2]
    y_mod=np.array(mod_labels),    # modulation labels
    y_sig=np.array(sig_labels),    # signal type labels
    y_snr=np.array(snrs)           # SNRs (optional, unused in current model)
)
# ------------------------------------------------------
# ✅ STEP 5: Print summary stats for verification
# ------------------------------------------------------
# These help verify class balance in your dataset
print("✅ Saved filtered dataset to radcom_preprocessed.npz")
print("📊 Mod class counts:", Counter(mod_labels))
print("📊 Sig class counts:", Counter(sig_labels))
