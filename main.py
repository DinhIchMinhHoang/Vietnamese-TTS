# main.py
import argparse
import os
import uuid
import tempfile
import threading
import requests

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
            ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
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

    demo.queue().launch(server_port=7860, share=True)

# ================
# FastAPI Server
# ================
def launch_api():
    app = FastAPI()
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
            if not reference_audio:
                return {"status": "error", "message": "No reference audio provided."}
            
            #speed
            speed = max(0.3, min(2.0, speed))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await reference_audio.read())
                ref_audio_path = tmp.name

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
    