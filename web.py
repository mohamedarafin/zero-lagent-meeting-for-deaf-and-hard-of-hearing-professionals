import asyncio
import json
import time
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

app = FastAPI(title="ClearContext Engine")

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 160  # 160ms chunk window for low-latency ingest
CHUNK_SIZE = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000))
SILENCE_ENERGY_THRESHOLD = 0.015  # Energy threshold to detect pauses
PAUSE_DURATION_TRIGGER = 0.6     # 600ms pause triggers a turn-taking slot

# Local offline model initialization
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

active_sockets: list[WebSocket] = []
audio_buffer = []
last_speech_time = time.time()
is_currently_speaking = False

@app.websocket("/stream")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_sockets.append(ws)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        active_sockets.remove(ws)

async def emit_event(payload: dict):
    serialized = json.dumps(payload)
    stale = []
    for ws in active_sockets:
        try:
            await ws.send_text(serialized)
        except Exception:
            stale.append(ws)
    for ws in stale:
        active_sockets.remove(ws)

def process_audio(indata, frames, time_info, status):
    global last_speech_time, is_currently_speaking, audio_buffer
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    samples = np.squeeze(indata).astype(np.float32)
    energy = float(np.sqrt(np.mean(samples ** 2)))
    current_time = time.time()

    if energy > SILENCE_ENERGY_THRESHOLD:
        last_speech_time = current_time
        audio_buffer.append(samples)
        
        # Once we gather ~480ms of audio, run low-latency speculative hypothesis
        if len(audio_buffer) >= 3:
            combined = np.concatenate(audio_buffer)
            segments, _ = model.transcribe(combined, beam_size=1, language="en")
            hypothesis = " ".join([s.text.strip() for s in segments if s.text.strip()])
            
            if hypothesis:
                loop.run_until_complete(emit_event({
                    "type": "caption",
                    "speaker": "Speaker 1",
                    "text": hypothesis,
                    "is_final": False,
                    "timestamp": round(current_time, 2)
                }))
            is_currently_speaking = True

    else:
        # Detect pauses to trigger conversational turn slot
        if is_currently_speaking and (current_time - last_speech_time) > PAUSE_DURATION_TRIGGER:
            loop.run_until_complete(emit_event({
                "type": "turn_cue",
                "message": "Pause detected — Natural window to speak",
                "timestamp": round(current_time, 2)
            }))
            is_currently_speaking = False
            audio_buffer = []

def start_audio_capture():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=process_audio, blocksize=CHUNK_SIZE):
        while True:
            sd.sleep(100)

if __name__ == "__main__":
    import threading
    import uvicorn
    threading.Thread(target=start_audio_capture, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8765)