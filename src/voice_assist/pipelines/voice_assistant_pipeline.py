import multiprocessing as mp
from voice_assist.transacription.auto_spliter import transcribe
from voice_assist.llm.ai_agent import AI_AGENT
from voice_assist.voice.voice_process import create_speech

from colorama import Fore, Style, init
import os, time, queue, threading, wave
import numpy as np
import sounddevice as sd
import re
import ollama

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))

init(autoreset=True)  # so colors reset automatically


# --- Timing decorator ---
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(Fore.CYAN + f"[Metrics] {func.__name__} took {elapsed:.3f} seconds")
        return result

    return wrapper


ai_agent = AI_AGENT()

# --- Apply decorator to your functions ---
transcribe = timed(transcribe)
ai_agent.query_agent = timed(ai_agent.query_agent)
create_speech = timed(create_speech)

# --- Audio playback queue ---
audio_queue = queue.Queue()


# --- Sentence splitter ---
def split_into_sentences(text: str):
    """Basic sentence splitter using regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


# --- Main loop ---
if __name__ == "__main__":
    print(Fore.YELLOW + "Starting voice assistant")

    while True:
        text = transcribe()

        if text.strip() == "":
            print(Fore.YELLOW + "No transcription available. Exiting.")
            break
        else:
            print(Fore.YELLOW + f"Transcribed text = {text}")

        buffer = ""
        sentence_endings = re.compile(r"([.!?])")

        messages = [
            {"role": "system", "content": "Keep responses concise."},
            {"role": "user", "content": f"{text}"},
        ]

        for part in ollama.chat(model="tinyllama", messages=messages, stream=True):
            chunk = part["message"]["content"]
            buffer += chunk

            # Look for sentences
            while True:
                match = sentence_endings.search(buffer)
                if not match:
                    break

                # Sentence goes up to the end of the matched punctuation
                end_index = match.end()
                sentence = buffer[:end_index].strip()
                buffer = buffer[end_index:].lstrip()  # remove used part from buffer

                if sentence:
                    print(sentence)
                    create_speech(sentence, debug=True)  # generate wav

    # Cleanup: wait for audio to finish, then stop worker
    audio_queue.join()
    audio_queue.put(None)
