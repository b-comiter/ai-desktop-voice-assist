import multiprocessing
import time
import select, sys
import sounddevice as sd
import numpy as np
import soundfile as sf 
from pathlib import Path
from datetime import datetime
from colorama import Fore, init
from voice_assist.llm.llm_stream import sentence_streamer
from voice_assist.transacription.transcriber import Transcriber
from voice_assist.voice.voice_synth import Vocalizer
from voice_assist.utils.context_manager import ContextManager


from voice_assist.transacription.transcriber import Transcriber
from voice_assist.transacription.config import TranscriberConfig

# Create a configuration object
transcriberConfig = TranscriberConfig(
    sample_rate=24000,
    channels=1,
    block_duration=0.1,
    silence_threshold=0.02,
    silence_duration=2.0,
    output_dir=Path("data/audio_input")
)

context_manager = ContextManager()
context_manager.clear_user("user")

SAMPLE_RATE = 16000
OUTPUT_DIR = Path("data/audio_input")
OUTPUT_DIR.mkdir(exist_ok=True)
LLM_MODEL = "llama3.1:8b"

init(autoreset=True)

# -------------------------------
# Processes
# -------------------------------

def transcriber_process(transcribe_queue, command_queue, playback_queue):
    transcriber = Transcriber(config=transcriberConfig)
    while True:
        text = transcriber.transcribe(agent_audio_buffer=playback_queue)
        transcribe_queue.put((text, time.time()))
        print(f"[Transcriber] Produced: {text}")

        if text.lower().strip() in ("stop.", "start."):
            command_queue.put(text.lower().strip())
            print(f"[Transcriber] Detected voice command: {text}")

        time.sleep(1)

def llm_process(transcribe_queue, llm_queue, llm_running):
    while True:
        if not llm_running["value"]:
            time.sleep(0.1)
            continue

        if not transcribe_queue.empty():
            try:
                text, timestamp = transcribe_queue.get_nowait()
                print(f"[LLM Process] Received: {text}")
                sentence_streamer(LLM_MODEL, "user", text, llm_queue)
            except:
                continue
        else:
            time.sleep(0.01)
            continue

def voice_synthesizer(llm_queue, audio_queue, llm_running, engine="kokoro"):
    vocalizer = Vocalizer(engine=engine)
    print(f"[Voice Synthesizer] Using engine: {engine}")
    i = 0
    while True:
        if not llm_running["value"]:
            time.sleep(0.1)
            continue

        try:
            response, timestamp = llm_queue.get()
            print(f"[Voice Synthesizer] Received: {response}")
        except:
            time.sleep(0.01)
            continue

        current_time = time.time()
        latency = current_time - timestamp
        print(f"[Voice Synthesizer] Synthesizing: {response} (Latency: {latency:.3f}s)")

        samples, samplerate = vocalizer.create_audio(response)
        audio_queue.put((samples, samplerate, timestamp))

        # Save to file for debugging
        print(Fore.RED + f"[Voice Synthesizer] {samplerate}, {len(samples)}")
        samples = samples.flatten()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_filename = OUTPUT_DIR / f"agent_output_{timestamp}_{i}.wav"
        sf.write(wav_filename, samples, samplerate)
        print(Fore.CYAN + f"💾 Saved processed mic input: {wav_filename}")
        i += 1

def audio_playback(audio_queue, playback_queue, llm_running):
    while True:
        if not llm_running["value"]:
            time.sleep(0.1)
            continue

        if not audio_queue.empty():
            try:
                samples, samplerate, timestamp = audio_queue.get_nowait()
            except:
                continue
        else:
            time.sleep(0.01)
            continue

        # Send a copy to playback_queue for echo subtraction
        playback_queue.put((samples, samplerate, timestamp))

        sd.play(samples, samplerate)
        sd.wait()
        print(f"[Audio Playback] Played audio (timestamp {timestamp:.3f})")

def flush_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except:
            break

# -------------------------------
# Pipeline
# -------------------------------
def run_pipeline():
    with multiprocessing.Manager() as manager:
        llm_running = manager.dict()
        llm_running["value"] = True

        transcribe_queue = manager.Queue()
        llm_queue = manager.Queue()
        audio_queue = manager.Queue()
        playback_queue = manager.Queue(maxsize=200)  # for sharing AI audio with transcriber
        command_queue = manager.Queue()
        # when launching processes

        engine_choice = "edge"  # or "kokoro"

        procs = [
            multiprocessing.Process(target=transcriber_process, args=(transcribe_queue, command_queue, playback_queue), daemon=True),
            multiprocessing.Process(target=llm_process, args=(transcribe_queue, llm_queue, llm_running)),
            multiprocessing.Process(target=voice_synthesizer, args=(llm_queue, audio_queue, llm_running, engine_choice)),
            multiprocessing.Process(target=audio_playback, args=(audio_queue, playback_queue, llm_running))
        ]

        for p in procs:
            p.start()

        try:
            while True:
                command = None
                if not command_queue.empty():
                    try:
                        command = command_queue.get_nowait()
                    except:
                        pass

                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input:
                        command = user_input

                if command == "stop.":
                    print("[Main] Stop command received.")
                    llm_running["value"] = False
                    flush_queue(transcribe_queue)
                    flush_queue(llm_queue)
                    flush_queue(audio_queue)
                    flush_queue(playback_queue)
                    print("[Main] Paused and queues cleared.")
                elif command == "start.":
                    llm_running["value"] = True
                    print("[Main] Resumed processing.")
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[Main] Exiting...")
            for p in procs:
                p.terminate()
                p.join()


if __name__ == "__main__":
    run_pipeline()
