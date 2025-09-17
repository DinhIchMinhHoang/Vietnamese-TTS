# Vietnamese-TTS

A high-quality Vietnamese text-to-speech system based on F5-TTS, enabling both local and API usage.

This project is a port and enhancement of [F5-TTS-Vietnamese-100h](https://huggingface.co/spaces/hynt/F5-TTS-Vietnamese-100h) and [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS).

## Features

- Convert Vietnamese text to natural-sounding speech.
- Local command-line interface and Gradio API server.
- Supports multiple usage modes: local inference, API, and batch processing.

## Requirements

- **Operating System:** Linux is recommended. If you are on Windows, use WSL2 or Ubuntu.
- **Python:** 3.10+
- **Packages:** See `requirements.txt`.

## Installation
#### 1. Clone the repository:

```bash
git clone https://github.com/DinhIchMinhHoang/Vietnamese-TTS.git
cd Vietnamese-TTS
```


#### 2. Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate       # Linux / WSL
# .venv\Scripts\activate        # Windows PowerShell
```


#### 2. Install dependencies:
```
pip install -r requirements.txt
```

#### ⚠️ Make sure to use Linux/WSL/Ubuntu if possible; some dependencies and audio tools may fail on plain Windows.

## First Use

#### Test the installation with a simple example:

```
python app.py
```

#### This will run a small TTS example and generate ``outputs/name.wav.``
## Local Usage
#### Command-Line
```
python f5_tts/infer/infer_cli.py --text "Xin chào các bạn!" --output outputs/hello.wav
```

``--text``: Text to synthesize

``--output``: Path to save generated audio

#### Gradio API Server
```
python server.py
```


Open ``http://127.0.0.1:7860`` in your browser.

Paste text and generate speech via the web interface.


## Credits

Project based on [F5-TTS-Vietnamese-100h](https://huggingface.co/spaces/hynt/F5-TTS-Vietnamese-100h)

Core model code adapted from [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)