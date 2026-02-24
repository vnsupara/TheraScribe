from transformers import pipeline
import torch

"""
  In order to access the model, go to https://huggingface.co/google/medasr 
  and accept the conditions. Make a HF_token and paste below.

"""
my_token = "Key from HuggingFace website" 

try:
    asr = pipeline(
        "automatic-speech-recognition",
        model="google/medasr",
        device=0 if torch.cuda.is_available() else -1,
        token=my_token  # Pass token directly here
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")

