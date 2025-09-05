import ollama
import re
from voice_assist.utils.context_manager import ContextManager
from voice_assist.tools.tools import extract_tool_call
from ollama._types import ChatResponse
import multiprocessing as mp
from colorama import Fore
import json

PROMPT = ""


class AI_AGENT:
    def __init__(
        self, user_id="user", model="tinyllama", context_file="data/context.json"
    ):
        self.context_manager = ContextManager(context_file)
        self.model = model

        if self.context_manager.is_user(user_id):
            pass
        else:
            self.context_manager.add_message(
                user_id=user_id, role="system", content=PROMPT
            )

    def tool_query(
        self, output_queue: mp.Queue, input: str, user_id="user", tools=None
    ):
        self.context_manager.add_message(user_id, "user", input)

        reesponse = ollama.chat(
            model=self.model,
            messages=self.context_manager.get_history(user_id),
            tools=tools,
        )
        print(reesponse)

        extract_tool_call(reesponse.message)

    def stream_query(
        self, output_queue: mp.Queue, input: str, user_id="user", tools=None
    ):
        self.context_manager.add_message(user_id, "user", input)

        full_content_response = ""
        buffer = ""

        for part in ollama.chat(
            model=self.model,
            messages=self.context_manager.get_history(user_id),
            tools=tools,
            stream=True,
        ):
            token = part["message"]["content"]
            buffer += token
            full_content_response += token
            print(token, end="", flush=True)  # still print tokens in real-time

            # Check if a sentence has ended
            sentences = re.split(r"([.!?])", buffer)  # keep punctuation
            while len(sentences) > 2:  # means we have at least one full sentence
                sentence = sentences[0] + sentences[1]
                output_queue.put(sentence.strip())
                sentences = sentences[2:]

            buffer = "".join(sentences)

        if buffer.strip():
            output_queue.put(buffer.strip())

        return full_content_response.strip()

    def query_agent(self, sender="user", incoming_text="Hello World!"):
        print(f"AI agent model: {self.model} is generating reply for {sender} \n")

        # Add the incoming email to conversation
        self.context_manager.add_message(sender, "user", incoming_text)
        return self._generate_reply(sender)

        # ===== GENERATE REPLY WITH OLLAMA =====

    def _generate_reply(self, sender):
        # Query Ollama with full history
        response: ChatResponse = ollama.chat(
            model=self.model, messages=self.context_manager.get_history(sender)
        )

        print(f"OLLAMA response: {response.message.content}\n")

        reply_text = response.message.content

        # Save assistant's reply to conversation
        self.context_manager.add_message(sender, "assistant", reply_text)

        return self._parse_model_output(reply_text)

    def _parse_model_output(self, text, full_output=False):
        """
        Extracts text inside <think> tags and returns both
        the think-text and the remaining text.
        """
        # Find all text inside <think> tags
        think_texts: str = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        # Remove all <think> blocks to get remaining text
        remaining_text: str = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL
        ).strip()
        if full_output:
            return {
                "think": think_texts,  # list of strings
                "reply": remaining_text,  # string
            }
        else:
            return remaining_text  # string
