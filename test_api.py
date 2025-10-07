import requests

API_URL = "http://localhost:8000/voice/synthesize"
VOICES_URL = "http://localhost:8000/voice/voices"

def get_voices():
    resp = requests.get(VOICES_URL)
    resp.raise_for_status()
    return resp.json()["voices"]

def synthesize(text, voice=None, output_filename=None, speed=1.0):
    data = {
        "text": text,
        "speed": str(speed),
    }
    if voice:
        data["voice"] = voice
    if output_filename:
        data["output_filename"] = output_filename
    resp = requests.post(API_URL, data=data)
    resp.raise_for_status()
    print("Response:", resp.json())
    return resp.json()

def main():
    print("Getting available voices...")
    voices = get_voices()
    print("Available voices:", voices)
    print("Synthesizing audio with first voice...")
    result = synthesize("xin chào!", voice=voices[0], output_filename="test.wav")
    print("Download link:", result.get("url"))

if __name__ == "__main__":
    main()
