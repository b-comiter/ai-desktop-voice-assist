import json
import re
import json
import subprocess
import platform


def open_app(app_name):
    """
    Example App Names:
    """
    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            subprocess.run(["open", "-a", app_name])
        elif system == "Windows":  # Windows
            subprocess.run(["start", "", app_name], shell=True)
        elif system == "Linux":  # Linux (GNOME/KDE)
            subprocess.run([app_name])
        else:
            print(f"Unsupported OS: {system}")
    except Exception as e:
        print(f"Error opening app: {e}")


def play_music():
    """Tell Spotify to play a song by name (macOS only)."""
    # todo currently mac only

    print("Playing music on spotify")
    subprocess.run(["osascript", "-e", 'tell application "Spotify" to play'])


TOOLS = {
    "open_app": open_app,
    "play_music": play_music,
    # "search_web": search_web,
}


def extract_tool_call(message):
    """
    Extracts tool call info from an Ollama message.
    Supports:
      - Structured tool_calls (like llama3.1)
      - JSON inside content (like phi4-mini)
    """

    # --- Case 1: Native tool_calls (llama3.1 style) ---
    if message.tool_calls:
        tool = message.tool_calls[0].function
        tool_name = tool.name

        if tool_name == "open_app":
            args = tool.arguments["app_name"]
            if tool_name in TOOLS:
                TOOLS[tool_name](args)
                return {"name": tool.name, "arguments": tool.arguments}
        else:
            if tool_name in TOOLS:
                TOOLS[tool_name]()
                return {"name": tool.name, "arguments": tool.arguments}

        print(f"Unknown tool: {tool_name}")

        return {"name": tool.name, "arguments": tool.arguments}

    # --- Case 2: JSON inside content (phi4-mini style) ---
    # todo fix later
    if message.content:
        # Strip markdown fences if present
        content = re.sub(
            r"^```json|```$", "", message.content.strip(), flags=re.MULTILINE
        ).strip()
        try:
            parsed = json.loads(content)
            if parsed.get("type") == "function":
                print("Tool Call", parsed["function"]["name"])

                if parsed["function"]["name"] == "open_app":
                    open_app(parsed["function"]["parameters"])

                return {
                    "name": parsed["function"]["name"],
                    "arguments": parsed["function"]["parameters"],
                }
        except json.JSONDecodeError:
            pass  # Not valid JSON, fall through

    return None


def clean_json_output(response: str) -> str:
    """
    Removes Markdown-style code fences (``` or ```json) from the response.
    """
    # Remove triple backticks with or without "json"
    cleaned = re.sub(
        r"^```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE | re.MULTILINE
    )
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    return cleaned


def detect_tool_call(response: str):
    """
    Detects if the AI agent's response is a tool call (JSON) or normal text.

    Returns:
        ("tool", dict) if it's a valid tool call
        ("text", str) if it's normal conversation
    """
    cleaned = clean_json_output(response)
    print(cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and ("tool" in parsed or "name" in parsed):
            return "tool", parsed
        else:
            return "text", response
    except json.JSONDecodeError:
        return "text", response


def open_app_from_json(json_data: dict):
    """
    Opens an application specified in a JSON tool call.
    Example JSON:
    {
        "name": "open_app",
        "parameters": {
            "appName": "Google Chrome"
        }
    }
    """
    system = platform.system()

    try:
        # Check if this is an open_app tool call
        if json_data.get("name") == "open_app":
            app_name = json_data["parameters"]["appName"]

            if system == "Darwin":  # macOS
                subprocess.run(["open", "-a", app_name])
            elif system == "Windows":  # Windows
                subprocess.run(["start", "", app_name], shell=True)
            elif system == "Linux":  # Linux (GNOME/KDE)
                subprocess.run([app_name])
            else:
                print(f"Unsupported OS: {system}")
        else:
            print("JSON is not an open_app command")
    except Exception as e:
        print(f"Error opening app: {e}")


# --- Example usage ---
resp = {"name": "open_app", "parameters": {"appName": "Google Chrome"}}

open_app_from_json(resp)


if __name__ == "__main__":
    play_music

    # --- Example usage ---
    resp1 = '{"tool": "search", "input": {"query": "latest news"}}'
    resp2 = "Ava: Sure, I can help you with that!"
    resp3 = """
            ```json
            {
            "name": "open_app",
            "parameters": {
                "appName": "Google Chrome"
            }
            }
            ```
    """
    resp4 = "```jsons"
    resp5 = """
        ```json
        {
            "type": "web_search",
            "query": "Football"
        }
        ```
"""
    subprocess.run(["osascript", "-e", 'tell application "Spotify" to play'])

    print(
        detect_tool_call(resp1)
    )  # -> ("tool", {'tool': 'search', 'input': {'query': 'latest news'}})
    print(
        detect_tool_call(resp2)
    )  # -> ("text", 'Ava: Sure, I can help you with that!')
    print(detect_tool_call(resp3.strip()))
    print(detect_tool_call(resp4))
    print(detect_tool_call(resp5.strip()))

    # Simulated llama3.1 message
    llama_message = type(
        "Message",
        (),
        {
            "tool_calls": [
                type(
                    "ToolCall",
                    (),
                    {
                        "function": type(
                            "Function",
                            (),
                            {"name": "open_app", "arguments": {"app_name": "Spotify"}},
                        )()
                    },
                )
            ],
            "content": None,
        },
    )()

    # Simulated phi4-mini message
    phi_message = type(
        "Message",
        (),
        {
            "tool_calls": None,
            "content": """```json
    {
    "type": "function",
    "function": {
        "name": "open_app",
        "parameters": {
        "app_name": "Spotify"
        }
    }
    }
    ```""",
        },
    )()

    print(extract_tool_call(llama_message))
    # ➝ {'name': 'open_app', 'arguments': {'app_name': 'Spotify'}}

    print(extract_tool_call(phi_message))
    # ➝ {'name': 'open_app', 'arguments': {'app_name': 'Spotify'}}
