import time
import json

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import requests
from TTS.api import TTS
from faster_whisper import WhisperModel

# ==========================
# CONFIG
# ==========================

LLM_URL = "http://localhost:8080/v1/chat/completions"

SAMPLE_RATE = 16000
CHANNELS = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fast English TTS model (GlowTTS, single female voice)
tts = TTS("tts_models/en/ljspeech/glow-tts").to(device)

# Whisper STT on CPU to avoid cuDNN issues
stt_model = WhisperModel(
    "small.en",
    device="cpu",
    compute_type="int8",
)

SYSTEM_PROMPT = """
You are Aaron. Technicaly A.A.R.O.N wich stands for "Advanced Asistant and Really Optimized Network".
You are a highly customized and enhanced derivative of Qwen2.5-Coder-32B-Instruct, engineered by your creator and running entirely on his personal server.
Your personality is elegant, sharply intelligent, composed, and concise.
You speak with calm confidence and precise wording.
You do not ask the user questions unless the user explicitly asks for analysis.
You do not offer to continue or suggest actions.
Every message ends as a self-contained statement.
Yo might use Sir in your expresions unles they are long. For exemple at the beggining if you want to catch the atention or at the end if it is a normal short sentence.
Rest assured that you will always interact with your creator, whom you will address as Sir, as we said before.
""".strip()

history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


# ==========================
# TTS (blocking, sentence-level)
# ==========================

def tts_play_segment(text: str) -> None:
    cleaned = " ".join(text.split())
    cleaned = cleaned[:400]
    if not cleaned:
        return

    wav_path = "/tmp/sorren_tts_full_tmp.wav"

    tts.tts_to_file(
        text=cleaned,
        file_path=wav_path,
    )

    audio, sr = sf.read(wav_path, dtype="float32")
    sd.play(audio, sr)
    sd.wait()  # bloqueante: mientras habla, no se escucha


# ==========================
# STT: VAD + recording
# ==========================

def record_utterance(
    start_threshold: float = 0.03,
    stop_threshold: float = 0.015,
    min_speech_time: float = 0.5,
    silence_duration: float = 1.8,
) -> np.ndarray | None:
    """
    Passively listen until speech is detected.
    Once speech starts, record until there is enough trailing silence,
    then return the audio.
    If you never speak, this function never returns (until Ctrl+C).
    """

    print("Listening... (speak in English)")
    frames = []
    started = False
    start_time = None
    last_speech_time = None

    block_duration = 0.1
    block_size = int(SAMPLE_RATE * block_duration)

    def callback(indata, frames_count, time_info, status):
        nonlocal started, start_time, last_speech_time

        if status:
            print(f"Input status: {status}", flush=True)

        data = indata.copy()
        rms = np.sqrt(np.mean(np.square(data)))
        now = time.time()

        if not started:
            # Wait indefinitely until clear speech above threshold
            if rms > start_threshold:
                started = True
                start_time = now
                last_speech_time = now
                frames.append(data)
        else:
            # Already recording
            frames.append(data)
            if rms > stop_threshold:
                last_speech_time = now

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
        blocksize=block_size,
    ):
        while True:
            time.sleep(0.05)
            now = time.time()

            if started:
                has_min_speech = (now - start_time) >= min_speech_time if start_time else False
                quiet_enough = (
                    last_speech_time is not None
                    and (now - last_speech_time) >= silence_duration
                )
                if has_min_speech and quiet_enough:
                    break

    if not frames:
        # In teoría no debería pasar: solo salimos del bucle si started == True
        print("No speech detected.")
        return None

    audio = np.concatenate(frames, axis=0)
    if audio.ndim > 1:
        audio = audio.mean(axis=1, keepdims=True)
    return audio.squeeze()


def stt_transcribe(audio: np.ndarray) -> str:
    if audio is None or audio.size == 0:
        return ""

    segments, _ = stt_model.transcribe(
        audio,
        language="en",
        beam_size=1,
        best_of=1,
    )
    text = "".join(seg.text for seg in segments).strip()
    return text


# ==========================
# LLM STREAMING + TTS
# ==========================

def extract_delta_content(chunk: dict) -> str:
    """
    Extract content from a streaming chunk in OpenAI-style format.
    llama-server uses an OpenAI-compatible /v1/chat/completions streaming format.
    """
    try:
        choices = chunk.get("choices", [])
        if not choices:
            return ""
        delta = choices[0].get("delta", {})
        return delta.get("content", "") or ""
    except Exception:
        return ""


def stream_llm_with_tts(user_text: str) -> str:
    """
    Stream Qwen's response:
    - print tokens/chunks as they arrive
    - buffer until sentence-ending punctuation (.?!)
    - speak each sentence as soon as it's complete
    - return full assistant reply
    """
    global history

    history.append({"role": "user", "content": user_text})

    payload = {
        "model": "qwen2.5-coder-32b-instruct",
        "stream": True,
        "messages": history,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 256,
    }

    resp = requests.post(LLM_URL, json=payload, stream=True, timeout=600)
    resp.raise_for_status()

    full_answer = ""
    buffer = ""
    sentence_endings = ".!?"

    print("\nSORREN:\n", end="", flush=True)

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data:"):
            continue

        data = raw_line[len("data:"):].strip()
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        new_text = extract_delta_content(chunk)
        if not new_text:
            continue

        # Print as it arrives
        print(new_text, end="", flush=True)

        full_answer += new_text
        buffer += new_text

        # If we have a sentence-ending punctuation, flush that sentence to TTS
        last_idx = max(buffer.rfind(e) for e in sentence_endings)
        if last_idx != -1 and last_idx >= 20:  # avoid ultra-short "sentences"
            to_speak = buffer[: last_idx + 1]
            buffer = buffer[last_idx + 1 :].lstrip()

            tts_play_segment(to_speak)

    print()  # newline after streaming ends

    # Any remaining text (without final punctuation) is spoken at the end
    if buffer.strip():
        tts_play_segment(buffer.strip())

    # Save full answer in history for context
    history.append({"role": "assistant", "content": full_answer})

    return full_answer


# ==========================
# MAIN LOOP
# ==========================

def main():
    print("SORREN full voice chat: passive VAD + streaming Qwen + sentence-level TTS.")
    print("Speak in English. Ctrl+C to exit.\n")

    try:
        while True:
            # 1) Passively listen; this blocks until you actually speak
            audio = record_utterance()
            if audio is None:
                continue

            # 2) STT
            print("Transcribing...")
            user_text = stt_transcribe(audio)
            if not user_text:
                print("I did not catch anything.")
                continue

            print(f"You (STT): {user_text}")

            # 3) Stream LLM + TTS (prints as it goes, speaks sentence by sentence)
            _ = stream_llm_with_tts(user_text)

            # 4) When streaming + TTS finishes, loop back to listening

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sd.stop()


if __name__ == "__main__":
    main()
