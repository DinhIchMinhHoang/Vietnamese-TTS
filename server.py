# server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from cached_path import cached_path
from f5_tts.api import F5TTS
import uvicorn
import tempfile
import os
import requests
import uuid
import torch

app = FastAPI()

# Create outputs directory for generated wavs
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve wavs via /static
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# Automatically download model and config if not present
CKPT_FILE = str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/model_last.pt"))
VOCAB_FILE = str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/config.json"))

# Initialize the TTS model
tts = F5TTS(
    ckpt_file=CKPT_FILE,
    vocab_file=VOCAB_FILE,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def get_ngrok_url():
    """Try to detect ngrok public URL from local API"""
    try:
        resp = requests.get("http://127.0.0.1:4040/api/tunnels").json()
        for tunnel in resp["tunnels"]:
            if tunnel["public_url"].startswith("https://"):
                return tunnel["public_url"]
    except Exception:
        return None
    return None

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    reference_audio: UploadFile = File(None),
    reference_audio_path: str = Form(None),
    reference_text: str = Form(""),
    output_filename: str = Form(None),
    speed: float = Form(1.0),
):
    try:
        # Case 1: File uploaded
        if reference_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await reference_audio.read())
                ref_audio_path = tmp.name
        # Case 2: Path provided
        elif reference_audio_path:
            ref_audio_path = reference_audio_path
        else:
            return {"status": "error", "message": "No reference audio provided."}

        # Generate unique filename if not provided
        if not output_filename:
            output_filename = f"{uuid.uuid4().hex}.wav"
        else:
            # Ensure it ends with .wav
            if not output_filename.lower().endswith(".wav"):
                output_filename += ".wav"

        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Clamp speed to [0.3, 2.0]
        speed = max(0.3, min(2.0, speed))

        # Run inference
        tts.infer(
            ref_file=ref_audio_path,
            ref_text=reference_text,
            gen_text=text,
            file_wave=output_path,
            speed=speed,
        )

        # Get public ngrok URL if available
        ngrok_url = get_ngrok_url()
        file_url = f"{ngrok_url}/static/{output_filename}" if ngrok_url else f"/static/{output_filename}"

        return JSONResponse(content={
            "status": "success",
            "url": file_url,
            "filename": output_filename
        })

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
