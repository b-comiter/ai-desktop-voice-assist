from faster_whisper import WhisperModel


class Transcriber:
    def __init__(
        self, model_name="distil-small.en", device="cpu", compute_type="float32"
    ):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def data_transcribe(self, data):  # Data can be wav file path or audio buffer
        """
        Reads a single audio file and returns the transcript
        """
        segments, _ = self.model.transcribe(data, chunk_length=30)
        transcript = ""
        for seg in segments:
            transcript += seg.text + " "
        return transcript.strip()
