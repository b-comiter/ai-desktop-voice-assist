from multiprocessing.util import debug
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import soundfile as sf
from colorama import Fore, init
from datetime import datetime
from .config import TranscriberConfig

init(autoreset=True)  # so colors reset automatically

class Transcriber:
    def __init__(
        self, config: TranscriberConfig, model_name="distil-small.en", device="cpu", compute_type="float32"
    ):
        self.config = config
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.config.output_dir.mkdir(exist_ok=True)  # Ensure output directory exists

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
        block_size = int(self.config.sample_rate * self.config.block_duration)
        silence_limit = int(self.config.silence_duration / self.config.block_duration)
        print(Fore.BLUE + f"Listening... speak into the microphone.")

        with sd.InputStream(samplerate=self.config.sample_rate, channels=self.config.channels, dtype="float32") as stream:
            return self._transcribe_audio_stream(block_size, silence_limit, stream)

    def _transcribe_audio_stream(self, block_size, silence_limit, stream):
        while True:
            audio_buffer = []
            silence_counter = 0
            recording = False

            # --- Combined loop for waiting and capturing speech ---
            while True:
                block, _ = stream.read(block_size)
                block = block.flatten()  # Flatten the block to 1D array

                volume = np.abs(block).mean()

                if volume > self.config.silence_threshold:
                    audio_buffer.append(block.copy())
                    recording = True
                    silence_counter = 0  # Reset silence counter when speech is detected
                elif recording:
                    silence_counter += 1

                if recording and silence_counter > silence_limit:
                    print(Fore.CYAN + "📝 Transcribing...")
                    audio_data = np.concatenate(audio_buffer)
                    output_text = self.transcribe_wav_file(audio_data)

                    # Save processed mic input for debugging
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wav_filename = self.config.output_dir / f"mic_clean_{timestamp}.wav"
                    sf.write(wav_filename, audio_data, self.config.sample_rate)
                    print(Fore.CYAN + f"💾 Saved processed mic input: {wav_filename}")

                    return output_text