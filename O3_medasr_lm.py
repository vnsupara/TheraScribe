#!/usr/bin/env python3
"""
  # OPTIMAL PIPELINE WITH LM
  python3 medasr_lm.py --model facebook/wav2vec2-base-960h --audio tests/trial_voi>

  # pipeline-only (fast)
  python3 medasr_with_lm.py --model google/medasr --audio sample.wav -o transcript.txt --verbose

  # public CTC models
  facebook/wav2vec2-base-960h
  facebook/wav2vec2-large-960h-lv60
"""
import argparse
import os
import re
import sys
import dataclasses
import torch

from transformers import AutoConfig, pipeline, AutoProcessor, AutoModelForCTC, AutoTokenizer
try:
    import pyctcdecode
except Exception:
    pyctcdecode = None
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

def clean_text(text: str) -> str:
    t = text.replace("</s>", " ").replace("<s>", " ")
    t = " ".join(t.split())
    t = re.sub(r"\s+([,?.!;:])", r"\1", t)
    t = re.sub(r"([.?!])([A-Za-z0-9])", r"\1 \2", t)
    return t.strip()

def _restore_text(text: str) -> str:
    return text.replace(" ", "").replace("#", " ").replace("</s>", "").strip()

class LasrCtcBeamSearchDecoder:
    def __init__(self, tokenizer: AutoTokenizer, kenlm_model_path=None, **kwargs):
        if pyctcdecode is None:
            raise RuntimeError("pyctcdecode not installed; install it to use LM decoding.")
        vocab = [None] * tokenizer.vocab_size
        for tok, idx in tokenizer.vocab.items():
            if idx < tokenizer.vocab_size:
                vocab[idx] = tok
        if any(v is None for v in vocab):
            raise RuntimeError("Tokenizer vocab contains None entries; cannot build decoder vocab.")
        vocab[0] = ""
        for i in range(1, len(vocab)):
            piece = vocab[i]
            if not piece.startswith("<") and not piece.endswith(">"):
                piece = "▁" + piece.replace("▁", "#")
            vocab[i] = piece
        self._decoder = pyctcdecode.build_ctcdecoder(vocab, kenlm_model_path, **kwargs)

    def decode_beams(self, logits, beam_width=8):
        beams = self._decoder.decode_beams(logits, beam_width=beam_width)
        return [dataclasses.replace(b, text=_restore_text(b.text)) for b in beams]

# inference functions
def run_pipeline(model_id, audio_path, device, chunk_length_s, stride_length_s, verbose=False):
    dev = 0 if (device is not None and device >= 0 and torch.cuda.is_available()) else -1
    if verbose:
        print(f"[pipeline] loading {model_id} on device {dev}")
    asr = pipeline("automatic-speech-recognition", model=model_id, device=dev)
    if verbose:
        print("[pipeline] running inference (chunking)...")
    res = asr(audio_path, chunk_length_s=chunk_length_s, stride_length_s=stride_length_s)
    return res.get("text", res.get("transcription", ""))

def run_ctc_with_lm(model_id, audio_path, device, lm_path, beam_width, verbose=False):
    if verbose:
        print("[ctc] loading processor and model:", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to("cpu" if device < 0 or not torch.cuda.is_available() else f"cuda:{device}")

    # read audio 
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
    except Exception:
        import librosa
        audio, sr = librosa.load(audio_path, sr=None)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    target_sr = getattr(processor, "sampling_rate", None)
    if target_sr is None:
        try:
            target_sr = processor.feature_extractor.sampling_rate
        except Exception:
            target_sr = 16000

    if sr != target_sr:
        try:
            import torchaudio
            tensor = torch.from_numpy(audio).unsqueeze(0)
            audio = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=target_sr).squeeze(0).numpy()
            sr = target_sr
        except Exception:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    device_str = "cpu" if device < 0 or not torch.cuda.is_available() else f"cuda:{device}"
    inputs = {k: v.to(device_str) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()[0]

    if lm_path:
        if pyctcdecode is None:
            raise RuntimeError("pyctcdecode not installed; cannot use LM decoding.")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        decoder = LasrCtcBeamSearchDecoder(tokenizer, kenlm_model_path=lm_path)
        beams = decoder.decode_beams(logits, beam_width=beam_width)
        best = beams[0].text if beams else ""
        return best
    else:
        import numpy as np
        pred_ids = logits.argmax(axis=-1)
        decoded = processor.batch_decode([pred_ids])[0]
        return decoded

def main():
    p = argparse.ArgumentParser(description="ASR with optional CTC+LM (medasr helper)")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--audio", required=True, help="Path to audio file")
    p.add_argument("--lm", default=None, help="Local kenlm file (.kenlm) to use")
    p.add_argument("--download-lm", action="store_true", help="If true, try to download lm_6.kenlm from model repo")
    p.add_argument("--device", type=int, default=-1, help="GPU device id (0) or -1 for CPU")
    p.add_argument("--chunk-length", type=float, default=20.0)
    p.add_argument("--stride-length", type=float, default=2.0)
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("-o", "--output", required=True, help="Output cleaned transcript file")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.audio):
        print("Audio not found:", args.audio, file=sys.stderr); sys.exit(2)

    lm_path = args.lm
    if args.download_lm:
        if hf_hub_download is None:
            print("huggingface_hub not available; cannot download LM.", file=sys.stderr)
        else:
            try:
                print("[lm] attempting to download lm_6.kenlm from model repo...")
                lm_path = hf_hub_download(args.model, filename="lm_6.kenlm")
                print("[lm] downloaded:", lm_path)
            except Exception as e:
                print("[lm] download failed:", e, file=sys.stderr)
                lm_path = args.lm  # fallback to provided path

    # check model type
    try:
        cfg = AutoConfig.from_pretrained(args.model)
        arch = getattr(cfg, "architectures", None)
        model_type = getattr(cfg, "model_type", None)
        if args.verbose:
            print("[model] model_type:", model_type, "architectures:", arch)
    except Exception as e:
        print("[model] warning: could not load config:", e, file=sys.stderr)
        cfg = None
        arch = None

    is_ctc = False
    if cfg is not None:
        # heuristics: architectures or model_type containing 'CTC' or 'Wav2Vec2' or 'Lasr' etc.
        arch_str = " ".join(arch) if arch else ""
        if ("CTC" in arch_str) or ("Wav2Vec2" in arch_str) or ("Lasr" in arch_str) or (model_type and "ctc" in model_type.lower()):
            is_ctc = True

    try:
        if is_ctc and lm_path:
            if args.verbose:
                print("[main] using CTC + LM decoding")
            raw = run_ctc_with_lm(args.model, args.audio, args.device, lm_path, args.beam_width, args.verbose)
        elif is_ctc and not lm_path:
            if args.verbose:
                print("[main] CTC model detected but no LM provided; using greedy CTC decode")
            raw = run_ctc_with_lm(args.model, args.audio, args.device, None, args.beam_width, args.verbose)
        else:
            if args.verbose:
                print("[main] using pipeline ASR (no LM)")
            raw = run_pipeline(args.model, args.audio, args.device, args.chunk_length, args.stride_length, args.verbose)
    except Exception as e:
        print("Error during inference:", e, file=sys.stderr); sys.exit(3)

    cleaned = clean_text(raw)
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(cleaned + "\n")
    except Exception as e:
        print("Error writing output:", e, file=sys.stderr); sys.exit(4)

    if args.verbose:
        print("Wrote cleaned transcript to:", args.output)
    else:
        print(args.output)

if __name__ == "__main__":
    main()

