import sounddevice as sd
import numpy as np
import soundfile as sf
from voice_assist.transacription.transcriber import Transcriber
from datetime import datetime
from colorama import Fore, init
from pathlib import Path

SAMPLE_RATE = 16000  # Whisper-friendly rate
CHANNELS = 1
BLOCK_DURATION = 0.1  # seconds per block (100ms)
SILENCE_THRESHOLD = 0.01  # adjust for your mic/background
SILENCE_DURATION = 1  # stop if silence lasts this long
transcriber = Transcriber()
init(autoreset=True)  # so colors reset automatically

OUTPUT_DIR = Path("data/audio_input")
OUTPUT_DIR.mkdir(exist_ok=True)


def transcribe(debug=False):
    block_size = int(SAMPLE_RATE * BLOCK_DURATION)
    silence_limit = int(SILENCE_DURATION / BLOCK_DURATION)
    print(Fore.BLUE + f"Listening... speak into the microphone. {transcriber}")

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32"
    ) as stream:
        while True:
            audio_buffer = []
            silence_counter = 0
            recording = False

            # --- Wait for speech ---
            while not recording:
                block, _ = stream.read(block_size)
                block = block.flatten()
                volume = np.abs(block).mean()

                if volume > SILENCE_THRESHOLD:
                    audio_buffer.append(block.copy())
                    recording = True

            # --- Capture speech until silence ---
            while recording:
                block, _ = stream.read(block_size)
                block = block.flatten()
                volume = np.abs(block).mean()
                audio_buffer.append(block.copy())

                if volume <= SILENCE_THRESHOLD:
                    silence_counter += 1
                else:
                    silence_counter = 0

                if silence_counter > silence_limit:
                    recording = False

                    # --- Run transcription ---
                    print(Fore.CYAN + "📝 Transcribing...")
                    audio_data = np.concatenate(audio_buffer)
                    output_text = transcriber.data_transcribe(audio_data)

                    if debug == True:
                        # --- Create timestamped filenames ---
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        wav_filename = f"{OUTPUT_DIR}/input_{timestamp}.wav"
                        txt_filename = f"{OUTPUT_DIR}/text_{timestamp}.txt"

                        sf.write(wav_filename, audio_data, SAMPLE_RATE)
                        print(Fore.CYAN + f"💾 Saved {wav_filename}")

                        # --- Save transcription ---
                        with open(txt_filename, "w", encoding="utf-8") as f:
                            f.write(output_text)
                        print(f"💾 Saved transcript to {txt_filename}")

                    return output_text


def transcribe_process(llm_queue=None):
    while True:
        text = transcribe()
        print(Fore.CYAN + f"Transcription: {text}")

        if llm_queue:
            llm_queue.put(text)


if __name__ == "__main__":
    transcribe_process(llm_queue=None)
