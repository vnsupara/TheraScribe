import gradio as gr
import soundfile as sf
import numpy as np
import os, time
import sys 

OUT_PATH = sys.argv[1]
os.makedirs("tests", exist_ok=True)

def _is_array_like(x):
    return hasattr(x, "ndim") or isinstance(x, (list, tuple, np.ndarray))

def save_audio(audio):
    if audio is None:
        return "No audio recorded."

    info = {"received_type": type(audio).__name__}

    try:
        if isinstance(audio, tuple) and len(audio) == 2:
            a0, a1 = audio[0], audio[1]
            if isinstance(a0, (int, float)) and _is_array_like(a1):
                sr = int(a0)
                arr = np.asarray(a1)
            elif isinstance(a1, (int, float)) and _is_array_like(a0):
                sr = int(a1)
                arr = np.asarray(a0)
            else:
                if _is_array_like(a0):
                    arr = np.asarray(a0); sr = 16000
                elif _is_array_like(a1):
                    arr = np.asarray(a1); sr = 16000
                else:
                    return f"Unsupported tuple contents. Debug: {info}"

            info.update({"shape": getattr(arr, "shape", None), "sr": sr})

        elif isinstance(audio, dict):
            if "array" in audio and "sample_rate" in audio:
                arr = np.asarray(audio["array"]); sr = int(audio["sample_rate"])
            elif "data" in audio and "sample_rate" in audio:
                arr = np.asarray(audio["data"]); sr = int(audio["sample_rate"])
            else:
                arr = None; sr = None
                for v in audio.values():
                    if _is_array_like(v) and arr is None:
                        arr = np.asarray(v)
                    if isinstance(v, (int, float)) and sr is None:
                        sr = int(v)
                if arr is None:
                    return f"Unsupported dict contents. Debug: {info}"
                if sr is None:
                    sr = 16000
            info.update({"shape": getattr(arr, "shape", None), "sr": sr})

        elif _is_array_like(audio):
            arr = np.asarray(audio)
            sr = 16000
            info.update({"shape": arr.shape, "sr": sr})

        else:
            return f"Unsupported audio type: {type(audio)}. Debug: {info}"

        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        if np.issubdtype(arr.dtype, np.integer):
            # assume 16-bit PCM
            arr = arr.astype("float32") / 32768.0
        else:
            arr = arr.astype("float32")

        sf.write(OUT_PATH, arr, sr)
        info.update({"saved": OUT_PATH, "timestamp": time.time()})
        return f"Saved {OUT_PATH}. Debug: {info}"

    except Exception as e:
        return f"Error processing audio: {e}. Debug: {info}"

with gr.Blocks() as demo:
    gr.Markdown("Record from your browser and save a WAV for MedASR")
    audio_comp = gr.Audio(label="Record : click the mic)", type="numpy")
    save_btn = gr.Button("Save to *.wav")
    status = gr.Textbox()
    save_btn.click(fn=save_audio, inputs=audio_comp, outputs=status)

demo.launch()
