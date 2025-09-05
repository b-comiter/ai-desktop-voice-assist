import multiprocessing as mp
from voice_assist.transacription.auto_spliter import transcribe
from voice_assist.llm.llm_stream import sentence_streamer, tool_streamer
from voice_assist.voice.voice_stream import generate_audio_process, audio_output_process
from voice_assist.tools.tools import TOOLS
from colorama import Fore, init
import os
import time

tools = list(TOOLS.values())

init(autoreset=True)  # so colors reset automatically

LLM_MODEL = "llama3.1:8b"  # "llama3.1:8b" #'phi4-mini:3.8b'

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def transcriber(text_queue: mp.Queue, audio_queue: mp.Queue, playback_active: mp.Event):
    while True:
        text = transcribe()

        if text.strip() == "":
            print(Fore.YELLOW + "No transcription available. Exiting.")
            break
        else:
            print(Fore.YELLOW + f"Transcribed text = {text}")

        if "open" in text.lower() or "search" in text.lower() or "play" in text.lower():
            print(Fore.MAGENTA + "Key word detected")
            tool_streamer(LLM_MODEL, "user", text, text_queue, tools=tools)
        else:
            sentence_streamer(LLM_MODEL, "user", text, text_queue)

        # --- Queue wait as you already had ---
        while (
            not text_queue.empty()
            or not audio_queue.empty()
            or playback_active.is_set()
        ):
            time.sleep(0.1)


def synth_transcriber(text_queue: mp.Queue, audio_queue: mp.Queue):
    while True:
        text = text_queue.get()
        generate_audio_process(text, audio_queue)


# --- Main loop ---
def run():
    print(Fore.YELLOW + "Starting voice assistant")

    # --- Audio playback queue ---
    text_queue = mp.Queue()
    audio_queue = mp.Queue()
    playback_active = mp.Event()

    p1 = mp.Process(
        target=transcriber, args=(text_queue, audio_queue, playback_active), daemon=True
    )
    p2 = mp.Process(
        target=synth_transcriber, args=(text_queue, audio_queue), daemon=True
    )
    p3 = mp.Process(
        target=audio_output_process, args=(audio_queue, playback_active), daemon=True
    )

    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
