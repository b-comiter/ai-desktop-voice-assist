from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from pathlib import Path

class TranscriberConfig(BaseModel):
    sample_rate: PositiveInt = Field(24000, description="Audio sample rate (Hz)")
    channels: PositiveInt = Field(1, description="Number of audio channels")
    block_duration: PositiveFloat = Field(0.1, description="Duration of each audio block (seconds)")
    silence_threshold: float = Field(0.01, ge=0, le=1, description="Threshold for detecting silence")
    silence_duration: PositiveFloat = Field(2.0, description="Duration of silence to stop recording (seconds)")
    output_dir: Path = Field(Path("data/audio_input"), description="Directory to save audio and transcripts")

    class Config:
        validate_assignment = True  # Automatically validate on assignment
