"""
Preprocess WAV files for MedASR:
- Ensures 16 kHz sample rate
- Ensures mono channel
- Converts to PCM_16 WAV
Usage:
    python audio_preprocess.py input.wav output.wav
If output path omitted, writes input_preprocessed.wav next to input.
"""

import os
import sys
import tempfile
import argparse
import soundfile as sf
import numpy as np

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except Exception:
    TORCHAUDIO_AVAILABLE = False
    try:
        import librosa
        LIBROSA_AVAILABLE = True
    except Exception:
        LIBROSA_AVAILABLE = False

try:
    import librosa.effects
    TRIM_AVAILABLE = True
except Exception:
    TRIM_AVAILABLE = False

TARGET_SR = 16000
TARGET_SUBTYPE = "PCM_16"


def read_wav(path):
    data, sr = sf.read(path, dtype="float32")
    return data, sr


def to_mono(arr):
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=1)


def resample_array(arr, orig_sr, target_sr=TARGET_SR):
    if orig_sr == target_sr:
        return arr
    if TORCHAUDIO_AVAILABLE:
        tensor = torch_from_numpy(arr)
        resampled = torchaudio.functional.resample(tensor, orig_freq=orig_sr, new_freq=target_sr)
        return numpy_from_torch(resampled)
    elif LIBROSA_AVAILABLE:
        return librosa.resample(arr, orig_sr, target_sr)
    else:
        raise RuntimeError(
            "No resampler available. Install torchaudio or librosa to enable resampling."
        )


def torch_from_numpy(arr):
    import torch
    if arr.ndim == 1:
        tensor = torch.from_numpy(arr).unsqueeze(0)
    else:
        tensor = torch.from_numpy(arr.T)  # shape (channels, samples)
    return tensor.float()


def numpy_from_torch(tensor):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr.squeeze(0)
    if arr.ndim == 2:
        return arr.T
    return arr


def normalize_pcm16(arr):
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype("float32") / 32768.0
    arr = np.clip(arr, -1.0, 1.0)
    return arr


def trim_silence(arr, sr, top_db=30):
    if not TRIM_AVAILABLE:
        return arr
    trimmed, _ = librosa.effects.trim(arr, top_db=top_db)
    return trimmed


def ensure_16k_mono(in_path, out_path=None, trim=False, top_db=30):
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_preprocessed.wav"

    data, sr = read_wav(in_path)
    data = to_mono(data)
    data = normalize_pcm16(data)
    if trim:
        data = trim_silence(data, sr, top_db=top_db)
    if sr != TARGET_SR:
        if TORCHAUDIO_AVAILABLE:
            import torch
            tensor = torch.from_numpy(data).unsqueeze(0)
            resampled = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=TARGET_SR)
            data = resampled.squeeze(0).cpu().numpy()
        elif LIBROSA_AVAILABLE:
            data = librosa.resample(data, sr, TARGET_SR)
        else:
            raise RuntimeError("No resampler available. Install torchaudio or librosa.")
        sr = TARGET_SR

    sf.write(out_path, data, TARGET_SR, subtype=TARGET_SUBTYPE)
    return out_path


def info(path):
    i = sf.info(path)
    return {
        "samplerate": i.samplerate,
        "channels": i.channels,
        "duration": i.duration,
        "format": i.format,
        "subtype": i.subtype,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess WAV for MedASR")
    parser.add_argument("input", help="Input WAV path")
    parser.add_argument("output", nargs="?", help="Output WAV path (optional)")
    parser.add_argument("--trim", action="store_true", help="Trim leading/trailing silence")
    parser.add_argument("--top_db", type=int, default=30, help="Silence trim threshold (dB)")
    args = parser.parse_args()

    inp = args.input
    out = args.output
    if not os.path.exists(inp):
        print("Input file not found:", inp)
        sys.exit(1)

    print("Input info:", info(inp))
    try:
        out_path = ensure_16k_mono(inp, out_path=out, trim=args.trim, top_db=args.top_db)
    except Exception as e:
        print("Error during preprocessing:", e)
        sys.exit(2)

    print("Wrote preprocessed file:", out_path)
    print("Output info:", info(out_path))


if __name__ == "__main__":
    main()
