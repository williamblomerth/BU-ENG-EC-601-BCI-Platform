import os
import numpy as np
import mne
import tensorflow as tf

# Import your EEGNet implementation
from EEG_Tensorflow_models.Models.EEGNet import EEGNet
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

###############################################################################
# USER CONFIGURATION
###############################################################################

DATASET_DIR = "/project/ece601/EEG ALS/EEGET-ALS Dataset/"   # <-- adjust path as needed
SAVE_MODEL_PATH = "eegnet_4ch_best_model.h5"

# Channels we want to keep
KEEP_CHANNELS = ["T7", "T8", "Fp1", "Fp2"]

# Training parameters
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def load_eeg_file(filepath):
    """Load a single EDF EEG file."""
    raw = mne.io.read_raw_edf(filepath, preload=True, stim_channel=None)
    return raw

def select_channels(raw, keep_channels):
    """Restrict the dataset to the desired channels."""
    picks = mne.pick_channels(raw.info["ch_names"], keep_channels, ordered=True)
    raw_reduced = raw.copy().pick(picks)
    return raw_reduced

def extract_epochs_and_labels(raw, tmin=0.0, tmax=1.0):
    """
    Extract epochs from annotations in EDF files.
    Returns:
        X: ndarray of shape (n_epochs, n_channels, n_times)
        y: ndarray of event labels
    """
    # Check if there are any annotations
    if not raw.annotations or len(raw.annotations) == 0:
        raise RuntimeError("No annotations found in EDF file.")
    
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True
    )
    
    X = epochs.get_data()
    y = epochs.events[:, -1]
    return X, y, event_id

def load_dataset(dataset_dir):
    """Recursively load all .edf files in the dataset directory."""
    X_list, y_list = [], []
    all_event_ids = {}

    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            if fn.endswith(".edf"):
                filepath = os.path.join(root, fn)
                print(f"Loading {filepath}...")
                raw = load_eeg_file(filepath)
                raw = select_channels(raw, KEEP_CHANNELS)
                
                try:
                    X, y, event_id = extract_epochs_and_labels(raw)
                    X_list.append(X)
                    y_list.append(y)
                    
                    # Merge event IDs across files
                    all_event_ids.update(event_id)
                except Exception as e:
                    print(f"Skipping {filepath}, could not extract epochs/labels: {e}")

    if not X_list:
        raise RuntimeError("No EEG data loaded. Check file paths, channels, and annotations.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y, all_event_ids

def map_labels_to_consecutive(y):
    """Map arbitrary integer labels to 0-indexed consecutive integers."""
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_mapped = np.array([label_map[val] for val in y])
    return y_mapped, label_map

###############################################################################
# MAIN
###############################################################################

def main():
    print("TensorFlow devices:", tf.config.list_physical_devices())

    print("Loading dataset...")
    X, y, event_id = load_dataset(DATASET_DIR)

    # Map labels to 0-indexed consecutive integers
    y, label_map = map_labels_to_consecutive(y)
    print("Label mapping (original -> new):", label_map)

    # EEGNet expects 4D input: (samples, channels, times, 1)
    X = X[:, :, :, np.newaxis]

    class_count = len(np.unique(y))
    y_cat = to_categorical(y, class_count)

    print("X shape:", X.shape)
    print("y shape:", y_cat.shape)

    # Build EEGNet
    model = EEGNet(
        nb_classes=class_count,
        Chans=X.shape[1],
        Samples=X.shape[2],
        dropoutRate=0.5,
        kernLength=64,
        F1=8,
        D=2,
        F2=16
    )

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    model.summary()

    # Train the model
    history = model.fit(
        X, y_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        shuffle=True
    )

    # Save the trained model
    print("Saving model to:", SAVE_MODEL_PATH)
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    main()
