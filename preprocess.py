"""
preprocess.py
=============
Fixed preprocessing - extracts only the sleep period (not pre/post wake).
"""

import os
import mne
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "data"
SAVE_DIR = "data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

SFREQ = 100
EPOCH_SEC = 30
SAMPLES_PER_EPOCH = SFREQ * EPOCH_SEC

MAX_SUBJECTS = 13

# Sleep stage mapping
ANNOTATION_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}

STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# How much Wake to include before first sleep and after last sleep
WAKE_BUFFER_MINUTES = 30  # Include 30 min of wake before/after sleep


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def bandpass_filter(data, sfreq=100, l_freq=0.5, h_freq=30.0):
    """Apply bandpass filter to EEG signal."""
    try:
        filtered = mne.filter.filter_data(
            data.astype(np.float64),
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            verbose=False
        )
        return filtered
    except:
        return data


def find_eeg_channel(raw):
    """Find the best EEG channel."""
    ch_names = raw.ch_names
    preferred = ["EEG Fpz-Cz", "EEG Pz-Oz"]

    for pref in preferred:
        for ch in ch_names:
            if pref.lower() in ch.lower():
                return ch

    for ch in ch_names:
        if "eeg" in ch.lower():
            return ch

    return None


def load_annotations(hyp_path):
    """Load sleep stage annotations."""
    try:
        annotations = mne.read_annotations(hyp_path)

        ann_list = []
        for onset, duration, description in zip(
            annotations.onset,
            annotations.duration,
            annotations.description
        ):
            if description in ANNOTATION_MAP:
                stage = ANNOTATION_MAP[description]
                if stage >= 0:
                    ann_list.append((float(onset), float(duration), int(stage)))

        return ann_list
    except Exception as e:
        print(f"    Annotation error: {e}")
        return []


def find_sleep_period(annotations):
    """
    Find the actual sleep period (first sleep to last sleep).
    Returns (start_time, end_time) in seconds.
    """
    sleep_stages = [1, 2, 3, 4]  # N1, N2, N3, REM

    first_sleep_time = None
    last_sleep_time = None

    for onset, duration, stage in annotations:
        if stage in sleep_stages:
            if first_sleep_time is None:
                first_sleep_time = onset
            last_sleep_time = onset + duration

    if first_sleep_time is None:
        return None, None

    # Add buffer for wake epochs around sleep
    buffer_sec = WAKE_BUFFER_MINUTES * 60
    start_time = max(0, first_sleep_time - buffer_sec)
    end_time = last_sleep_time + buffer_sec

    return start_time, end_time


def extract_sleep_epochs(eeg_data, annotations, sfreq, recording_duration):
    """
    Extract epochs only from the sleep period.
    """
    # Find sleep period boundaries
    sleep_start, sleep_end = find_sleep_period(annotations)

    if sleep_start is None:
        print(f"    No sleep detected in annotations")
        return None, None

    print(f"    Sleep period: {sleep_start/3600:.2f}h - {sleep_end/3600:.2f}h")

    X_list = []
    y_list = []
    total_samples = len(eeg_data)

    for onset, duration, stage in annotations:
        # Skip annotations outside sleep period
        if onset + duration < sleep_start:
            continue
        if onset > sleep_end:
            continue

        # Clip to sleep period
        effective_onset = max(onset, sleep_start)
        effective_end = min(onset + duration, sleep_end)
        effective_duration = effective_end - effective_onset

        if effective_duration < EPOCH_SEC:
            continue

        # Number of epochs in this segment
        n_epochs = int(effective_duration // EPOCH_SEC)

        for i in range(n_epochs):
            epoch_start_sec = effective_onset + (i * EPOCH_SEC)
            start_idx = int(epoch_start_sec * sfreq)
            end_idx = start_idx + SAMPLES_PER_EPOCH

            if start_idx < 0 or end_idx > total_samples:
                continue

            epoch = eeg_data[start_idx:end_idx]

            if len(epoch) != SAMPLES_PER_EPOCH:
                continue

            if np.isnan(epoch).any() or np.isinf(epoch).any():
                continue
            if np.std(epoch) < 1e-10:
                continue

            X_list.append(epoch)
            y_list.append(stage)

    if len(X_list) == 0:
        return None, None

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def process_recording(psg_path, hyp_path):
    """Process a single PSG recording."""
    try:
        # Load PSG
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

        eeg_channel = find_eeg_channel(raw)
        if eeg_channel is None:
            print(f"    No EEG channel found")
            return None, None

        print(f"    Using channel: {eeg_channel}")

        raw.pick([eeg_channel])

        if raw.info['sfreq'] != SFREQ:
            raw.resample(SFREQ, verbose=False)

        eeg_data = raw.get_data()[0]
        recording_duration = len(eeg_data) / SFREQ
        print(f"    Recording: {recording_duration/3600:.2f} hours")

        # Preprocess
        eeg_data = bandpass_filter(eeg_data, sfreq=SFREQ)
        mean_val = np.mean(eeg_data)
        std_val = np.std(eeg_data)

        if std_val < 1e-10:
            print(f"    Zero variance")
            return None, None

        eeg_data = (eeg_data - mean_val) / std_val

        # Load annotations
        annotations = load_annotations(hyp_path)

        if len(annotations) == 0:
            print(f"    No annotations")
            return None, None

        print(f"    Annotations: {len(annotations)} segments")

        # Extract only sleep period
        X, y = extract_sleep_epochs(eeg_data, annotations, SFREQ, recording_duration)

        if X is None or len(X) < 10:
            print(f"    Too few epochs")
            return None, None

        return X, y

    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def find_file_pairs(data_dir):
    """Find matching PSG and Hypnogram file pairs."""
    pairs = []
    search_dirs = [data_dir]

    for subdir in ["sleep-cassette", "sleep-telemetry"]:
        subpath = os.path.join(data_dir, subdir)
        if os.path.exists(subpath):
            search_dirs.append(subpath)

    print(f"Searching: {search_dirs}")

    for search_dir in search_dirs:
        try:
            files = os.listdir(search_dir)
        except:
            continue

        psg_files = [f for f in files if f.endswith("-PSG.edf")]
        hyp_files = [f for f in files if "Hypnogram.edf" in f]

        psg_map = {f[:6]: os.path.join(search_dir, f) for f in psg_files}
        hyp_map = {f[:6]: os.path.join(search_dir, f) for f in hyp_files}

        matched = set(psg_map.keys()) & set(hyp_map.keys())
        print(f"  {search_dir}: {len(matched)} pairs")

        for sid in sorted(matched):
            pairs.append((sid, psg_map[sid], hyp_map[sid]))

    return pairs


def main():
    print("=" * 70)
    print("SLEEP-EDF PREPROCESSING (SLEEP PERIOD ONLY)")
    print("=" * 70)

    pairs = find_file_pairs(DATA_DIR)
    print(f"\nFound {len(pairs)} recording pairs")

    if len(pairs) == 0:
        print("ERROR: No data found!")
        return

    if MAX_SUBJECTS:
        pairs = pairs[:MAX_SUBJECTS]

    X_all, y_all = [], []
    processed, skipped = 0, 0

    print("\n" + "-" * 70)

    for sid, psg_path, hyp_path in tqdm(pairs, desc="Processing"):
        print(f"\n[{sid}]")

        X, y = process_recording(psg_path, hyp_path)

        if X is None:
            skipped += 1
            continue

        unique, counts = np.unique(y, return_counts=True)
        dist = ", ".join([f"{STAGE_NAMES[u]}:{c}" for u, c in zip(unique, counts)])
        print(f"    Extracted {len(y)} epochs: {dist}")

        X_all.append(X)
        y_all.append(y)
        processed += 1

    if len(X_all) == 0:
        print("\nERROR: No data extracted!")
        return

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    np.save(os.path.join(SAVE_DIR, "X.npy"), X)
    np.save(os.path.join(SAVE_DIR, "y.npy"), y)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Processed: {processed} | Skipped: {skipped}")
    print(f"Total epochs: {len(y)}")
    print(f"Shape: {X.shape}")
    print()

    print("Class Distribution:")
    print("-" * 40)
    unique, counts = np.unique(y, return_counts=True)
    for stage, count in zip(unique, counts):
        pct = 100.0 * count / len(y)
        bar = "█" * int(pct / 2)
        print(f"  {STAGE_NAMES[stage]:>4}: {count:6d} ({pct:5.1f}%) {bar}")

    print(f"\nSaved to: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
