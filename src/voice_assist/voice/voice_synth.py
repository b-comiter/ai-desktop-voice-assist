from kokoro_onnx import Kokoro


class Vocalizer:
    def __init__(self, language="en"):
        self.model = Kokoro(
            "src/voice_assist/voice/kokoro/kokoro-v1.0.onnx",
            "src/voice_assist/voice/kokoro/voices-v1.0.bin",
        )

    def create_audio(self, text):
        samples, sample_rate = self.model.create(
            text, voice="af_bella", speed=1.25, lang="en-us"
        )
        return samples, sample_rate


if __name__ == "__main__":
    vocalizer = Vocalizer()
