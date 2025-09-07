from multiprocessing.util import debug
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import soundfile as sf
from colorama import Fore, init
from pathlib import Path
from datetime import datetime

SAMPLE_RATE = 24000  # Whisper-friendly rate
CHANNELS = 1
BLOCK_DURATION = 0.1  # seconds per block (100ms)
SILENCE_THRESHOLD = 0.01  # adjust for your mic/background
SILENCE_DURATION = 2  # stop if silence lasts this long
init(autoreset=True)  # so colors reset automatically

OUTPUT_DIR = Path("data/audio_input")
OUTPUT_DIR.mkdir(exist_ok=True)

class Transcriber:
    def __init__(
        self, model_name="distil-small.en", device="cpu", compute_type="float32"
    ):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe_wav_file(self, data):  # Data can be wav file path or audio buffer
        """
        Reads a single audio file and returns the transcript
        """
        segments, _ = self.model.transcribe(data, chunk_length=30)
        transcript = ""
        for seg in segments:
            transcript += seg.text + " "
        return transcript.strip()
    
    def transcribe(self, debug=False, agent_audio_buffer=None):
        block_size = int(SAMPLE_RATE * BLOCK_DURATION)
        silence_limit = int(SILENCE_DURATION / BLOCK_DURATION)
        print(Fore.BLUE + f"Listening... speak into the microphone.")

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
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

                    audio_buffer.append(block.copy())

                    volume = np.abs(block).mean()
                    if volume <= SILENCE_THRESHOLD:
                        silence_counter += 1
                    else:
                        silence_counter = 0

                    if silence_counter > silence_limit:
                        recording = False
                        print(Fore.CYAN + "📝 Transcribing...")

                        audio_data = np.concatenate(audio_buffer)
                        output_text = self.transcribe_wav_file(audio_data)

                        # Save processed mic input for debugging
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        wav_filename = OUTPUT_DIR / f"mic_clean_{timestamp}.wav"
                        sf.write(wav_filename, np.array(audio_data, dtype=np.float32), SAMPLE_RATE)
                        print(Fore.CYAN + f"💾 Saved processed mic input: {wav_filename}")

                        if debug:
                            txt_filename = f"{OUTPUT_DIR}/text_{timestamp}.txt"
                            with open(txt_filename, "w", encoding="utf-8") as f:
                                f.write(output_text)
                            print(f"💾 Saved transcript to {txt_filename}")

                        return output_text