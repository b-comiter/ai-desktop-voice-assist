import multiprocessing as mp
import re
import ollama
from voice_assist.llm.ai_agent import AI_AGENT


def sentence_streamer(
    model: str, user_id, input, text_queue, tools=None
) -> None:
    ai_agent = AI_AGENT(user_id=user_id, model=model)
    """
    Stream tokens from Ollama LLM and enqueue complete sentences.
    Each sentence is pushed to the queue as soon as it's complete.
    """

    try:
        full_response = ai_agent.stream_query(text_queue, input, user_id=user_id, tools=tools)
        ai_agent.context_manager.add_message("user", "assistant", full_response)
    finally:
        # Sentinel to tell the consumer that streaming is done
        text_queue.put(None)


def tool_streamer(model: str, user_id, input, text_queue: mp.Queue, tools=None) -> None:
    ai_agent = AI_AGENT(user_id=user_id, model=model)
    """
    Stream tokens from Ollama LLM and enqueue complete sentences.
    Each sentence is pushed to the queue as soon as it's complete.
    """
    try:
        full_response = ai_agent.tool_query(
            text_queue, input, user_id=user_id, tools=tools
        )
        ai_agent.context_manager.add_message("user", "assistant", full_response)
    finally:
        # Sentinel to tell the consumer that streaming is done
        text_queue.put(None)


def sentence_consumer(text_queue: mp.Queue) -> None:
    """
    Consume sentences from the queue and handle them.
    Replace the print() with whatever processing you want.
    """
    while True:
        sentence = text_queue.get()
        if sentence is None:
            print("\n[Info] Sentence consumer shutting down.")
            break
        print(f"\n[Sentence ready] {sentence}")


if __name__ == "__main__":
    model = "llama3.2:1b"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story about black holes in space."},
    ]

    # Create a multiprocessing queue
    text_queue = mp.Queue()

    # Start consumer process
    consumer_process = mp.Process(target=sentence_consumer, args=(text_queue,))
    consumer_process.start()

    # Run the LLM streamer in the main process
    sentence_streamer(
        model, "user", "tell me a story about black holes in space.", text_queue, []
    )

    # Wait for consumer to finish
    consumer_process.join()
