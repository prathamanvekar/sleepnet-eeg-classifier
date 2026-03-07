import argparse
import json
import os
from pathlib import Path

import numpy as np


STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def _band_limited_noise(rng, n, sfreq, f_lo, f_hi, scale=1.0):
    """Quick-and-dirty band-limited noise via FFT masking."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    X[~mask] = 0
    y = np.fft.irfft(X, n=n)
    y = y / (np.std(y) + 1e-12)
    return y * scale


def _make_epoch_synthetic(stage, rng, sfreq=100, epoch_sec=30):
    n = int(sfreq * epoch_sec)
    t = np.arange(n) / sfreq

    # baseline drift (very low freq) + broadband noise
    drift = 0.15 * np.sin(2 * np.pi * rng.uniform(0.05, 0.25) * t)
    broadband = 0.25 * rng.standard_normal(n)

    if stage == 0:  # W: alpha/beta stronger
        alpha = 0.9 * np.sin(2 * np.pi * rng.uniform(8, 12) * t)
        beta = 0.4 * np.sin(2 * np.pi * rng.uniform(16, 25) * t)
        sig = alpha + beta + drift + broadband
    elif stage == 1:  # N1: theta-ish, lower amplitude
        theta = 0.8 * np.sin(2 * np.pi * rng.uniform(4, 7) * t)
        sig = 0.7 * theta + drift + 0.3 * broadband
    elif stage == 2:  # N2: spindles + some theta; occasional K-complex
        theta = 0.4 * np.sin(2 * np.pi * rng.uniform(4, 7) * t)
        spindle = np.sin(2 * np.pi * rng.uniform(12, 14) * t)
        # spindle bursts
        env = np.zeros(n)
        for _ in range(rng.integers(1, 4)):
            start = rng.integers(0, n - int(1.0 * sfreq))
            dur = rng.integers(int(0.6 * sfreq), int(1.5 * sfreq))
            end = min(n, start + dur)
            w = np.hanning(end - start)
            env[start:end] += w
        spindles = 1.0 * spindle * env

        # crude K-complex (single sharp negative-positive deflection)
        k = np.zeros(n)
        if rng.random() < 0.4:
            center = rng.integers(int(2 * sfreq), n - int(2 * sfreq))
            width = rng.integers(int(0.4 * sfreq), int(1.0 * sfreq))
            win = np.hanning(width)
            pulse = -2.0 * win
            k[center : center + width] += pulse

        sig = theta + spindles + k + drift + 0.25 * broadband
    elif stage == 3:  # N3: strong delta, higher amplitude
        delta = 1.6 * np.sin(2 * np.pi * rng.uniform(0.8, 2.5) * t)
        slow = _band_limited_noise(rng, n, sfreq, 0.5, 4.0, scale=1.2)
        sig = delta + 0.7 * slow + drift + 0.15 * broadband
    else:  # REM: mixed freq, low-ish amplitude, more “noisy”
        mixed = (
            0.35 * np.sin(2 * np.pi * rng.uniform(4, 8) * t)
            + 0.25 * np.sin(2 * np.pi * rng.uniform(10, 18) * t)
        )
        sig = mixed + drift + 0.45 * broadband

    # normalize each epoch (roughly mimic preprocessing stability)
    sig = sig - np.mean(sig)
    sig = sig / (np.std(sig) + 1e-12)
    # scale to “microvolt-ish” magnitude (arbitrary; frontend doesn’t care)
    sig = sig * rng.uniform(15e-6, 60e-6)
    return sig.astype(np.float64)


def make_from_processed(x_path, y_path, out_csv, n_epochs=60, seed=0, shuffle=False):
    X = np.load(x_path)  # (N, 3000)
    y = np.load(y_path) if y_path and os.path.exists(y_path) else None

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D (epochs, samples). Got shape={X.shape}")
    if X.shape[1] != 3000:
        raise ValueError(f"Expected 3000 samples/epoch. Got {X.shape[1]}")

    rng = np.random.default_rng(seed)

    n_epochs = int(min(n_epochs, X.shape[0]))
    if shuffle:
        idx = rng.choice(X.shape[0], size=n_epochs, replace=False)
    else:
        # grab a random contiguous window (feels more “real” as a hypnogram)
        start = int(rng.integers(0, X.shape[0] - n_epochs + 1))
        idx = np.arange(start, start + n_epochs)

    stream = X[idx].reshape(-1)

    # tiny jitter/noise so uploads aren’t identical every time
    stream = stream.astype(np.float64)
    stream += (0.01 * np.std(stream) + 1e-12) * rng.standard_normal(stream.size)

    np.savetxt(out_csv, stream, delimiter=",")
    meta = {
        "mode": "from_processed",
        "x_path": str(x_path),
        "y_path": str(y_path) if y_path else None,
        "n_epochs": int(n_epochs),
        "samples": int(stream.size),
        "sfreq": 100,
        "epoch_sec": 30,
        "samples_per_epoch": 3000,
        "shuffle": bool(shuffle),
        "seed": int(seed),
    }
    if y is not None:
        yy = y[idx].tolist()
        meta["stages"] = yy
        meta["stage_names"] = STAGE_NAMES
        meta["stage_counts"] = {STAGE_NAMES[i]: int(sum(np.array(yy) == i)) for i in range(5)}

    meta_path = str(Path(out_csv).with_suffix(".meta.json"))
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def make_synthetic(out_csv, n_epochs=60, seed=0, sfreq=100, epoch_sec=30):
    rng = np.random.default_rng(seed)

    # simple hypnogram-ish progression (not medical-grade, just UI testing)
    # Start with W, go down, bounce, end with REM
    stages = []
    stages += [0] * rng.integers(2, 6)  # W
    stages += [1] * rng.integers(1, 3)  # N1
    stages += [2] * rng.integers(8, 15)  # N2
    stages += [3] * rng.integers(4, 10)  # N3
    stages += [2] * rng.integers(6, 12)  # back to N2
    stages += [4] * rng.integers(3, 8)  # REM
    # repeat pattern until reach n_epochs
    while len(stages) < n_epochs:
        stages += [2] * rng.integers(6, 12)
        stages += [4] * rng.integers(3, 8)
        stages += [0] * rng.integers(1, 4)
    stages = stages[:n_epochs]

    epochs = [_make_epoch_synthetic(s, rng, sfreq=sfreq, epoch_sec=epoch_sec) for s in stages]
    stream = np.concatenate(epochs, axis=0)

    np.savetxt(out_csv, stream.astype(np.float64), delimiter=",")

    meta = {
        "mode": "synthetic",
        "n_epochs": int(n_epochs),
        "samples": int(stream.size),
        "sfreq": int(sfreq),
        "epoch_sec": int(epoch_sec),
        "samples_per_epoch": int(sfreq * epoch_sec),
        "seed": int(seed),
        "stages": stages,
        "stage_names": STAGE_NAMES,
        "stage_counts": {STAGE_NAMES[i]: int(sum(np.array(stages) == i)) for i in range(5)},
    }
    meta_path = str(Path(out_csv).with_suffix(".meta.json"))
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def main():
    p = argparse.ArgumentParser(description="Generate uploadable 1-column CSV for the web app.")
    p.add_argument("--mode", choices=["from_processed", "synthetic"], default="from_processed")
    p.add_argument("--out", default="sample_eeg.csv")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle", action="store_true", help="Only for from_processed")
    p.add_argument("--x", default=os.path.join("data", "processed", "X.npy"))
    p.add_argument("--y", default=os.path.join("data", "processed", "y.npy"))
    args = p.parse_args()

    out_csv = args.out

    if args.mode == "from_processed":
        if not os.path.exists(args.x):
            raise FileNotFoundError(f"Missing {args.x}. Run preprocess.py first, or use --mode synthetic.")
        meta_path = make_from_processed(args.x, args.y, out_csv, n_epochs=args.epochs, seed=args.seed, shuffle=args.shuffle)
    else:
        meta_path = make_synthetic(out_csv, n_epochs=args.epochs, seed=args.seed)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {meta_path}")
    print("Upload the CSV (single-column) in the browser and click Analyze.")


if __name__ == "__main__":
    main()
