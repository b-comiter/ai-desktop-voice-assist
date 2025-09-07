import soundfile as sf
from voice_assist.voice.voice_synth import Vocalizer
import sounddevice as sd
from colorama import Fore
import time

def synthesize_speech(text, vocalizer, debug=False):
    """Generate audio for text using a given Vocalizer instance."""
    if debug:
        start = time.time()
    print(Fore.GREEN + "Synthesizing_Speech")
    samples, sample_rate = vocalizer.create_audio(text)

    if debug:
        print(
            Fore.GREEN
            + f"Synthesis took: {time.time() - start} using Vocalizer {vocalizer}"
        )

    return samples, sample_rate


def play_audio(samples, sample_rate):
    sd.play(samples, sample_rate)
    sd.wait()  # Wait until the audio is finished playing


def save_audio_to_file(samples, sample_rate, filename="output.wav"):
    sf.write(filename, samples, sample_rate)
    print(f"Audio saved to {filename}")


def create_speech(text, vocalizer, debug=False):
    data, sample_rate = synthesize_speech(text, vocalizer, debug)
    play_audio(data, sample_rate)


def speech_process(speech_queue):
    """Example speech process that instantiates a single Vocalizer per process."""
    vocalizer = Vocalizer()  # Only once per process
    print(vocalizer, "Vocalizer object created for this process")
    last_end_time = None  # track when last audio finished

    while True:
        text = speech_queue.get()  # blocking until next text
        if text is None:  # Exit signal
            break

        # Clean text
        text = "".join(
            c for c in text if c.isalnum() or c.isspace() or c in ".,!?;:'\"-"
        )
        print(f"[Speech] {text}")

        start_wait = time.time()
        data, sample_rate = synthesize_speech(text, vocalizer)
        wait_duration = time.time() - start_wait
        print(f"[Metrics] Synthesis took {wait_duration:.3f}s")

        # Measure playback duration
        start_play = time.time()
        if last_end_time is not None:
            idle_time = start_play - last_end_time
            print(f"[Metrics] Idle time before next speech: {idle_time:.3f}s")

        play_audio(data, sample_rate)
        end_play = time.time()
        play_duration = end_play - start_play
        print(f"[Metrics] Audio played | Duration: {play_duration:.3f}s")

        last_end_time = end_play


if __name__ == "__main__":
    text = """
    Black holes are regions of space-time where the gravitational force is so strong that even light cannot escape from them.
In other words, they are areas where the mass of a star or object is so large that it has no effect on the gravity of the surrounding area.
Black holes are thought to be at the end of all stars and galaxies, and they are also known as "dark matter" because their effects have yet to be directly detected.
Black holes can form through the interactions between two massive objects or through the collapse of a star due to its own mass.
In some cases, these events are predicted by theoretical models.
The gravity of a black hole is so strong that anything passing into it is pulled inwards towards its core, forming a pointlike object that appears to be infinite in size.
Black holes are also thought to play a role in the formation of new stars and galaxies.
They can trap gas and dust around them, preventing it from escaping and causing the stars to form.
In some cases, black holes can even act as magnets, attracting other objects towards themselves while repelling other objects away.
Black holes are considered one of the most complex phenomena in the universe, and their study is a major area of research in astrophysics and particle physics.
Many theories and models have been proposed to explain how black holes form and interact with matter, and current scientific understanding of these phenomena continues to evolve as new discoveries are made."""

    data, sample_rate = synthesize_speech(text, debug=True)
    play_audio(data, sample_rate)
