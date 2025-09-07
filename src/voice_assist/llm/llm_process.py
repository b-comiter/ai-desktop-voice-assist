import ollama
import re
from colorama import Fore
from voice_assist.llm.ai_agent import AI_AGENT

DEBUG = False
DEBUG_EXTENDED = False

def llm_process(llm_queue, speech_queue=None):
    ai_agent = AI_AGENT()
    rolling_transcript = ""
    print("🤖 LLM process started...")

    while True:
        text_chunk = llm_queue.get()
        rolling_transcript += " " + text_chunk
        rolling_transcript = rolling_transcript.strip()

        # Skip processing if text chunk is empty or just whitespace
        if not text_chunk or text_chunk.isspace() or DEBUG:
            print(
                f"Received empty or whitespace-only chunk, skipping LLM call {text_chunk}."
            )
            continue
        elif DEBUG_EXTENDED:
            print(f"Received text chunk: '{text_chunk}'")

        ai_response = ai_agent.query_agent(incoming_text=text_chunk)

        if speech_queue:
            speech_queue.put(ai_response)


if __name__ == "__main__":
    ai_agent = AI_AGENT(model="qwen3:0.6b")
    response = ai_agent.query_agent()


def parse_model_output(text, full_output=False) -> dict:
    """
    Extracts text inside <think> tags and returns both
    the think-text and the remaining text.
    """
    # Find all text inside <think> tags
    think_texts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    print("wat")
    # Remove all <think> blocks to get remaining text
    remaining_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    print("wat2")

    if full_output:
        return {
            "think": think_texts,  # list of strings
            "reply": remaining_text,  # string
        }
    else:
        return remaining_text  # string
