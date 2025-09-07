import multiprocessing as mp
from voice_assist.voice.voice_process import synthesize_speech, play_audio
from colorama import Fore

def generate_audio_process(text: str, audio_buffer_queue: mp.Queue, debug= False) -> None:
    """
    Generate audio from text and put the buffer into the queue.
    """
    if text is None:
        print(Fore.GREEN + f"[Info] Reached end of stream.")
        audio_buffer_queue.put(None)
        return
    
    try:
        audio_buffer, sample_rate = synthesize_speech(text, debug)
        audio_buffer_queue.put((audio_buffer, sample_rate))
    except Exception as e:
        print(Fore.RED + f"[Error] Failed to synthesize speech: {e}")

def audio_output_process(audio_queue: mp.Queue, playback_active: mp.Event) -> None:
    """
    Continuously play audio buffers from the queue.
    Terminates when a sentinel value (None) is received.
    """
    while True:
        item = audio_queue.get()  # blocking
        
        if item != None:
            try:
                if playback_active:
                    playback_active.set()   # 🔴 signal: audio is playing
                audio_buffer, sample_rate = item
                play_audio(audio_buffer, sample_rate)

            except Exception as e:
                print(Fore.RED + f"[Error] Failed to play audio: {e}")
        else:
            print("Playback Status ", playback_active)
            playback_active.clear()
            print("Playback Status ", playback_active)

    print(Fore.RED + "[Info] Audio output process terminating gracefully.")
    return


if __name__ == "__main__":
    # Example text
    text = ["""
                Black holes are regions of space-time where the gravitational force
                is so strong that even light cannot escape from them.
            """,
            """
                In other words, they are areas where the mass of a star or object is so large that it has no effect on the gravity of the surrounding area.
                Black holes are thought to be at the end of all stars and galaxies, and they are also known as "dark matter" because their effects have yet to be directly detected.
                Black holes can form through the interactions between two massive objects or through the collapse of a star due to its own mass.
            """,
            """
                In some cases, these events are predicted by theoretical models.
            """, 
            """
                Black holes are also thought to play a role in the formation of new stars and galaxies.
                They can trap gas and dust around them, preventing it from escaping and causing the stars to form.
                In some cases, black holes can even act as magnets, attracting other objects towards themselves while repelling other objects away.
                Black holes are considered one of the most complex phenomena in the universe, and their study is a major area of research in astrophysics and particle physics.
                Many theories and models have been proposed to explain how black holes form and interact with matter, and current scientific understanding of these phenomena continues to evolve as new discoveries are made.
            """]

    # Create multiprocessing queue
    audio_queue = mp.Queue()
    playback_active = mp.Event()

    # Start audio output process
    audio_process = mp.Process(target=audio_output_process, args=(audio_queue, playback_active))
    audio_process.start()

    # Generate audio and enqueue it
    for sentence in text:
        generate_audio_process(sentence, audio_queue, debug= True)

    # Signal the process to exit gracefully
    audio_queue.put(None)
    audio_process.join()
