# main.py
import argparse
import os
import uuid
import tempfile
import threading
import requests
import glob

import torch
import uvicorn
import gradio as gr
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

    # Save audio if path provided
    if output_path:
        import soundfile as sf
        sf.write(output_path, final_wave, final_sample_rate)

    return final_sample_rate, final_wave, spectrogram

# ================
# Gradio UI
# ================
def launch_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis.
        Upload a sample voice and enter text to generate natural speech.
        """)

        with gr.Row():
            ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath", value="example_voice/sample.wav")
            gen_text = gr.Textbox(label="📝 Text", placeholder="Enter text...", lines=3)

        speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Speed")
        btn_synthesize = gr.Button("🔥 Generate Voice")

        with gr.Row():
            output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
            output_spectrogram = gr.Image(label="📊 Spectrogram")

        def infer_ui(ref_audio, gen_text, speed):
            if not ref_audio:
                raise gr.Error("Please upload a sample audio file.")
            if not gen_text.strip():
                raise gr.Error("Please enter text to generate voice.")
            sr, wave, spec = run_inference(ref_audio, "", gen_text, speed)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                save_spectrogram(spec, tmp.name)
                spectrogram_path = tmp.name
            return (sr, wave), spectrogram_path

        btn_synthesize.click(infer_ui, inputs=[ref_audio, gen_text, speed], outputs=[output_audio, output_spectrogram])

    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)

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
# FastAPI Server
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
        speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡️ Speed")
        btn_synthesize = gr.Button("🔥 Generate Voice")

        with gr.Row():
            output_audio = gr.Audio(label="Generated Audio", type="numpy")
            output_spectrogram = gr.Image(label="Spectrogram")

        def infer_ui(selected_voice, ref_audio_path, gen_text, speed):
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

            sr, wave, spec = run_inference(resolved_ref, "", gen_text.strip(), speed)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                save_spectrogram(spec, tmp.name)
                spectrogram_path = tmp.name
            return (sr, wave), spectrogram_path

        btn_synthesize.click(
            infer_ui,
            inputs=[voice_dropdown, ref_audio, gen_text, speed],
            outputs=[output_audio, output_spectrogram]
        )

    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)

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
    