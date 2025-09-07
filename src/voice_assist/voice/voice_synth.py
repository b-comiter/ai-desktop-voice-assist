import asyncio
import tempfile
import os
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from edge_tts import Communicate
import sounddevice as sd


class Vocalizer:
    def __init__(
        self,
        engine="kokoro",  # "kokoro" or "edge"
        kokoro_model_path="src/voice_assist/voice/kokoro/kokoro-v1.0.onnx",
        kokoro_voices_path="src/voice_assist/voice/kokoro/voices-v1.0.bin",
        kokoro_voice="af_bella",
        kokoro_speed=1.25,
        edge_voice="en-US-AvaNeural"
    ):
        self.kokoro_model_path = kokoro_model_path
        self.kokoro_voices_path = kokoro_voices_path
        self.kokoro_voice = kokoro_voice
        self.kokoro_speed = kokoro_speed
        self.edge_voice = edge_voice

        self.engine = None
        self.model = None
        self.use(engine)  # initialize the chosen engine

    def use(self, engine: str):
        """Switch between 'kokoro' and 'edge' at runtime."""
        engine = engine.lower()
        self.engine = engine

        if engine == "kokoro":
            self.model = Kokoro(self.kokoro_model_path, self.kokoro_voices_path)
            print(f"[Vocalizer] Switched to Kokoro (voice={self.kokoro_voice}, speed={self.kokoro_speed})")
        elif engine == "edge":
            self.model = None  # Edge TTS doesn’t preload a model
            print(f"[Vocalizer] Switched to Edge TTS (voice={self.edge_voice})")
        else:
            raise ValueError("engine must be 'kokoro' or 'edge'")

    def create_audio(self, text):
        if self.engine == "kokoro":
            samples, sample_rate = self.model.create(
                text,
                voice=self.kokoro_voice,
                speed=self.kokoro_speed,
                lang="en-us"
            )
            return samples, sample_rate

        elif self.engine == "edge":
            return self._edge_tts_to_buffer(text)

        else:
            raise RuntimeError("Engine not set. Call use('kokoro') or use('edge').")

    def _edge_tts_to_buffer(self, text):
        """Run Edge TTS → return (samples, sample_rate) without writing permanent files."""
        tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmpfile.close()  # let Edge TTS write into it
        asyncio.run(self._edge_tts_to_file(text, tmpfile.name))

        data, sample_rate = sf.read(tmpfile.name, dtype='int16')
        os.remove(tmpfile.name)

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.int16)

        return data, sample_rate

    async def _edge_tts_to_file(self, text, filename):
        communicate = Communicate(text, self.edge_voice)
        with open(filename, "wb") as out:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    out.write(chunk["data"])


if __name__ == "__main__":
    vocalizer = Vocalizer(engine="kokoro", kokoro_voice="af_bella")
    audio, sample_rate = vocalizer.create_audio("Hello from Kokoro! This is a test.")
    print(f"Kokoro: {audio.shape}, {sample_rate} Hz")

    sd.play(audio, sample_rate)
    sd.wait()

    # Switch to Edge dynamically
    vocalizer.use("edge")
    audio, sample_rate = vocalizer.create_audio("Hello from Edge TTS! This is a test.")
    print(f"Edge: {audio.shape}, {sample_rate} Hz")

    sd.play(audio, sample_rate)
    sd.wait()

    # Switch back to Kokoro
    vocalizer.use("kokoro")
    audio, sample_rate = vocalizer.create_audio("Back to Kokoro now. This is a test.")
    print(f"Kokoro again: {audio.shape}, {sample_rate} Hz")

    sd.play(audio, sample_rate)
    sd.wait()
