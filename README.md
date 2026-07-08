# TheraScribe
A fully local, privacy-preserving ABA therapy pipeline using MedASR for transcription.

## Overview
ABA technicians often struggle to document sessions while simultaneously delivering instruction. THe pipeline automates that burden by taking recordings of session and transcribing speech using MedASR, finetuning it with CTC decoding, and running it thorugh MedGemma for analysis.

## Pipeline
1. Install dependencies
2. Clean audio
3. Transcribe with MedASR
4. Clean transcripts


## Installation
1. Install Pytorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
2. Install dependencies
   pip install -r requirements.txt
3. Load MedASR model (requirems HF-Token from https://huggingface.co/google/medasr)
   python3 medasr_pipeline/O0_install_model.py

## MedASR Pipeline
1. Load audio or record
   python3 audio_recording/O1_record_gradio.py
2. Clean audio: applies VAD, noise reudction, and normalization
   python3 audio_recording/O2_preprocess_wav.py input.wav output.wav
3. Run MedASR: uses CTC+LM decoding for improved accuracy
   python3 medasr_pipeline/O3_medasr_lm.py  --model facebook/wav2vec2-base-960h --audio recording.wav -o output.txt --verbose
4. Clean Transcript: restores punctuation, normalizes ABA terminology, and prepares text for MedGemma
   python3 medasr_pipeline/O4_clean_transcript.py input.txt -o output.txt





