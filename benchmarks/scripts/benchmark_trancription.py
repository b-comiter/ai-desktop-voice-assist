import time
from pathlib import Path
import csv
import soundfile as sf
from voice_assist.transacription.transcriber import Transcriber

# --- CONFIG ---
MODEL_SIZES = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "distil-small.en",
    "medium",
    "medium.en",
]  # list of model sizes to test

AUDIO_DIR = Path("data/samples")  # folder containing audio files to test
OUTPUT_DIR = Path("benchmarks/results")  # folder to save CSV
AUDIO_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_RUNS = 1  # average over multiple runs
MAKE_CSV = True  # whether to save results to CSV

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- HELPER FUNCTION ---
def benchmark_audio(transcriber: Transcriber, file_path: Path) -> float:
    start = time.time()
    segments = transcriber.file_transcribe(file_path)
    end = time.time()
    duration = end - start
    return duration, segments


# --- HELPER FUNCTIONS ---
def get_wav_duration(file_path: Path) -> float:
    """Return duration of a WAV file in seconds using soundfile."""
    with sf.SoundFile(file_path) as f:
        frames = len(f)
        rate = f.samplerate
        duration = frames / rate
    return duration


# --- RUN BENCHMARK ---
def main():
    benchmark_start = time.time()
    audio_files = list(AUDIO_DIR.glob("*.wav"))
    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")
        return

    print(f"Found {len(audio_files)} audio files in {AUDIO_DIR}")

    all_results = []

    for audio_file in audio_files:
        print(f"\n=== Benchmarking file: {audio_file.name} ===")
        wav_duration = get_wav_duration(audio_file)

        for model_size in MODEL_SIZES:
            transcriber = Transcriber(model_name=model_size)
            times = []
            for _ in range(NUM_RUNS):
                duration, text = benchmark_audio(transcriber, audio_file)
                times.append(duration)
            avg_time = sum(times) / NUM_RUNS
            print(f"{model_size}: {avg_time:.2f}s (WAV duration: {wav_duration:.2f}s)")
            all_results.append((audio_file.name, model_size, avg_time))

    # Save results to CSV
    csv_file = OUTPUT_DIR / "transcription_benchmark.csv"
    if MAKE_CSV:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "model_size", "avg_time_s"])
            writer.writerows(all_results)

    print(f"\nBenchmarking complete! Results saved to {csv_file}")

    test_duration = time.time() - benchmark_start
    print(
        f"Total benchmark runtime: {(test_duration):.2f} seconds ({test_duration / 60:.2f} minutes)"
    )


if __name__ == "__main__":
    main()
