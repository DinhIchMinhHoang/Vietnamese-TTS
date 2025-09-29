# main.py
import argparse
import os
import uuid
import tempfile
import numpy as np
import librosa
import threading
import requests
import glob
import scipy.signal
import parselmouth

import torch
import uvicorn
import gradio as gr
from pydub import AudioSegment, effects
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from cached_path import cached_path
from huggingface_hub import login

from vinorm import TTSnorm
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_vocoder,
    load_model,
    infer_process,
    save_spectrogram,
)

# ================
# Hugging Face login (if token is available)
# ================
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    login(token=hf_token)

# ================
# Shared Model Setup
# ================
VOCAB_FILE = str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/config.json"))
CKPT_FILE = str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/model_last.pt"))

vocoder = load_vocoder()
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=CKPT_FILE,
    vocab_file=VOCAB_FILE,
)

# ================
# Advanced Audio Effects Helpers
# ================
# 1. EQ/filtering (low-pass filter example)
def lowpass_filter(data, sr, cutoff=3000):
    b, a = scipy.signal.butter(N=6, Wn=cutoff/(sr/2), btype='low')
    return scipy.signal.lfilter(b, a, data)

# 2. Reverb (simple echo)
def add_echo(data, sr, delay=0.2, decay=0.4):
    delay_samples = int(delay * sr)
    echo = np.zeros_like(data)
    if delay_samples < len(data):
        echo[delay_samples:] = data[:-delay_samples] * decay
    return np.clip(data + echo, -1.0, 1.0)

# 3. Dynamic range compression (using pydub)
try:
    def compress_dynamic_range_np(data, sr):
        audio = AudioSegment(
            (data * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        compressed = effects.compress_dynamic_range(audio)
        return np.array(compressed.get_array_of_samples()).astype(np.float32) / 32767
except ImportError:
    def compress_dynamic_range_np(data, sr):
        raise ImportError("pydub is not installed. Run 'pip install pydub'.")

# 4. Silence insertion
def add_silence(data, sr, duration=0.5):
    silence = np.zeros(int(sr * duration), dtype=data.dtype)
    return np.concatenate([data, silence])

# 5. Pitch range modulation (vibrato-like effect)
def modulate_pitch(data, sr, depth=2, rate=5):
    """Apply vibrato-like pitch modulation. depth in semitones, rate in Hz."""
    t = np.arange(len(data)) / sr
    mod = depth * np.sin(2 * np.pi * rate * t)
    out = np.zeros_like(data)
    frame_size = 2048
    for i in range(0, len(data), frame_size):
        n_steps = mod[i]
        frame = data[i:i+frame_size]
        if len(frame) == 0:
            continue
        out[i:i+frame_size] = librosa.effects.pitch_shift(frame.astype(np.float32), sr=sr, n_steps=n_steps)
    return out

# 6. Formant shifting using praat-parselmouth
try:
    def shift_formants(data, sr, ratio=1.2):
        """Shift formants using praat-parselmouth. ratio > 1 raises formants, < 1 lowers."""
        snd = parselmouth.Sound(data, sr)
        manipulated = snd.clone().change_gender(formant_shift_ratio=ratio)
        return manipulated.values[0]
except ImportError:
    def shift_formants(data, sr, ratio=1.2):
        raise ImportError("parselmouth is not installed. Run 'pip install praat-parselmouth'.")

# ================
# Helpers
# ================
def post_process(text: str) -> str:
    """Clean up text for TTS."""
    text = " " + text + " "
    text = text.replace(" . . ", " . ").replace(" .. ", " . ")
    text = text.replace(" , , ", " , ").replace(" ,, ", " , ")
    text = text.replace('"', "")
    return " ".join(text.split())

def get_ngrok_url():
    """Try to detect ngrok public URL (if running)."""
    try:
        resp = requests.get("http://127.0.0.1:4040/api/tunnels").json()
        for tunnel in resp["tunnels"]:
            if tunnel["public_url"].startswith("https://"):
                return tunnel["public_url"]
    except Exception:
        return None
    return None

# ================
# Unified Inference
# ================
def run_inference(ref_audio_path, ref_text, gen_text, speed=1.0, output_path=None):
    """Main inference logic, shared by API + UI."""
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text or "")
    final_wave, final_sample_rate, spectrogram = infer_process(
        ref_audio,
        ref_text.lower(),
        post_process(TTSnorm(gen_text)).lower(),
        model,
        vocoder,
        speed=speed,
    )

    # These will be set by the UI, default to 0 if not present
    pitch_shift = getattr(run_inference, '_pitch_shift', 0)
    gain_db = getattr(run_inference, '_gain_db', 0)

    # Pitch shift (in semitones)
    if pitch_shift != 0:
        final_wave = librosa.effects.pitch_shift(final_wave.astype(np.float32), sr=final_sample_rate, n_steps=pitch_shift)

    # Gain (in dB)
    if gain_db != 0:
        factor = 10 ** (gain_db / 20)
        final_wave = final_wave * factor

    # Clip to [-1, 1] to avoid distortion
    final_wave = np.clip(final_wave, -1.0, 1.0)

    # Save audio if path provided
    if output_path:
        import soundfile as sf
        sf.write(output_path, final_wave, final_sample_rate)

    return final_sample_rate, final_wave, spectrogram

# ================
# List Default Voices
# ================
def _list_default_voices(voices_dir: str = "default_voices"):
    supported_exts = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg")
    files = []
    for pattern in supported_exts:
        files.extend(glob.glob(os.path.join(voices_dir, pattern)))
    files = sorted(files)
    # Map nice labels to absolute paths
    label_to_path = {os.path.basename(path): path for path in files}
    return list(label_to_path.keys()), label_to_path

# ================
# Gradio UI
# ================
def launch_ui():
    voice_labels, label_to_path = _list_default_voices("example_voice")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # F5-TTS: Vietnamese Text-to-Speech Synthesis.
        Chọn giọng có sẵn hoặc tải lên mẫu giọng, sau đó nhập văn bản để tạo giọng nói tự nhiên.
        """)

        with gr.Row():
            voice_dropdown = gr.Dropdown(
                choices=voice_labels,
                value=voice_labels[0] if voice_labels else None,
                label="Chọn giọng có sẵn (default_voices)",
                interactive=True
            )
            ref_audio = gr.Audio(
                label="Hoặc tải lên mẫu giọng (tùy chọn)",
                type="filepath"
            )

        gen_text = gr.Textbox(label="Text", placeholder="Enter text...", lines=3)

        with gr.Row():
            speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡️ Speed")
            pitch_shift = gr.Slider(-12, 12, value=0, step=1, label="🎵 Pitch Shift (semitones)")
            gain_db = gr.Slider(-20, 20, value=0, step=1, label="🔊 Gain (dB)")

        with gr.Accordion("Advanced Audio Effects", open=False):
            eq_cutoff = gr.Slider(500, 8000, value=3000, step=100, label="EQ Low-pass Cutoff (Hz)")
            echo_delay = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Echo Delay (s)")
            echo_decay = gr.Slider(0.0, 0.9, value=0.0, step=0.05, label="Echo Decay")
            compression = gr.Checkbox(label="Enable Compression", value=False)
            silence_duration = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="Silence Duration (s)")
            pitch_mod_depth = gr.Slider(0.0, 6.0, value=0.0, step=0.1, label="Pitch Mod Depth (semitones)")
            pitch_mod_rate = gr.Slider(1.0, 10.0, value=5.0, step=0.1, label="Pitch Mod Rate (Hz)")
            formant_ratio = gr.Slider(1.0, 1.0, value=1.0, step=0.01, label="Formant Shift Ratio")

        btn_synthesize = gr.Button("🔥 Generate Voice")

        with gr.Row():
            output_audio = gr.Audio(label="Generated Audio", type="numpy")
            output_spectrogram = gr.Image(label="Spectrogram")

        def infer_ui(selected_voice, ref_audio_path, gen_text, speed, pitch_shift_val, gain_db_val,
                    eq_cutoff_val, echo_delay_val, echo_decay_val, compression_enabled,
                    silence_duration_val, pitch_mod_depth_val, pitch_mod_rate_val, formant_ratio_val):
            # Resolve reference audio path: uploaded > selected default voice
            resolved_ref = None
            if ref_audio_path:
                resolved_ref = ref_audio_path
            elif selected_voice:
                path = label_to_path.get(selected_voice)
                if path and os.path.exists(path):
                    resolved_ref = path

            if not resolved_ref:
                raise gr.Error("Vui lòng chọn giọng có sẵn hoặc tải lên file audio mẫu.")
            if not gen_text or not gen_text.strip():
                raise gr.Error("Vui lòng nhập nội dung văn bản để tổng hợp giọng.")

            # Set attributes for pitch/gain for this call
            run_inference._pitch_shift = pitch_shift_val
            run_inference._gain_db = gain_db_val
            sr, wave, spec = run_inference(resolved_ref, "", gen_text.strip(), speed)
            # Clean up after call
            del run_inference._pitch_shift
            del run_inference._gain_db

            # Only apply effects if user changes from default (no adjustment)
            # 1. EQ/filtering
            if eq_cutoff_val != 3000:
                wave = lowpass_filter(wave, sr, cutoff=eq_cutoff_val)
            # 2. Echo/reverb
            if echo_delay_val > 0 and echo_decay_val > 0:
                wave = add_echo(wave, sr, delay=echo_delay_val, decay=echo_decay_val)
            # 3. Compression
            if compression_enabled:
                try:
                    wave = compress_dynamic_range_np(wave, sr)
                except Exception:
                    pass
            # 4. Silence insertion
            if silence_duration_val > 0:
                wave = add_silence(wave, sr, duration=silence_duration_val)
            # 5. Pitch modulation
            if pitch_mod_depth_val > 0:
                wave = modulate_pitch(wave, sr, depth=pitch_mod_depth_val, rate=pitch_mod_rate_val)
            # 6. Formant shifting
            if abs(formant_ratio_val - 1.0) > 0.01:
                try:
                    wave = shift_formants(wave, sr, ratio=formant_ratio_val)
                except Exception:
                    pass

            # Clip to [-1, 1] after all effects
            wave = np.clip(wave, -1.0, 1.0)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                save_spectrogram(spec, tmp.name)
                spectrogram_path = tmp.name
            return (sr, wave), spectrogram_path

        btn_synthesize.click(
            infer_ui,
            inputs=[voice_dropdown, ref_audio, gen_text, speed, pitch_shift, gain_db,
                    eq_cutoff, echo_delay, echo_decay, compression, silence_duration,
                    pitch_mod_depth, pitch_mod_rate, formant_ratio],
            outputs=[output_audio, output_spectrogram]
        )

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        root_path="/voice"
    )

# ================
# FastAPI Server
# ================
def launch_api():
    app = FastAPI(root_path="/voice")
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

    @app.post("/synthesize")
    async def synthesize(
        text: str = Form(...),
        reference_audio: UploadFile = File(None),
        reference_text: str = Form(""),
        output_filename: str = Form(None),
        speed: float = Form(1.0),
    ):
        try:

            # Use uploaded file or default sample
            if reference_audio:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(await reference_audio.read())
                    ref_audio_path = tmp.name
            else:
                ref_audio_path = "example_voice/sample.wav"

            #speed
            speed = max(0.3, min(2.0, speed))

            if not output_filename:
                output_filename = f"{uuid.uuid4().hex}.wav"
            elif not output_filename.lower().endswith(".wav"):
                output_filename += ".wav"

            output_path = os.path.join(OUTPUT_DIR, output_filename)

            run_inference(ref_audio_path, reference_text, text, speed, output_path)

            ngrok_url = get_ngrok_url()
            file_url = f"{ngrok_url}/static/{output_filename}" if ngrok_url else f"/static/{output_filename}"

            return JSONResponse(content={
                "status": "success",
                "url": file_url,
                "filename": output_filename
            })
        except Exception as e:
            return {"status": "error", "message": str(e)}

    uvicorn.run(app, host="0.0.0.0", port=8000)

# ================
# Run Both (UI + API)
# ================
def launch_both():
    threading.Thread(target=launch_api, daemon=True).start()
    launch_ui()

# ================
# Entry Point
# ================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ui", "api", "both"], default="ui")
    args = parser.parse_args()

    if args.mode == "ui":
        launch_ui()
    elif args.mode == "api":
        launch_api()
    elif args.mode == "both":
        launch_both()
