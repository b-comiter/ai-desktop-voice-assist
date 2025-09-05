# Desktop Voice Assistant

A local AI-powered voice assistant that lets you interact naturally with your computer. Speak to it, and it listens, thinks, and responds — just like a personal AI agent running on your machine.

## ✨ Features

* **Voice Interaction**: Speak naturally to the assistant, and get spoken responses back.
* **Local Processing**: Uses Ollama and other local tools — your data stays on your device.
* **Computer Control**: The assistant can open apps, play music, and more using JSON-based commands.
* **Benchmarking**: Built-in testing framework to compare different models and measure performance.

## 🛠 System Overview

The system is composed of three main parts:

1. **Transcriber**: Converts your voice to text in real-time.
2. **LLM Agent**: Processes text input, decides how to respond, and can issue JSON commands.
3. **Voice Synthesizer**: Converts responses back into natural-sounding speech.

This flow makes the assistant function like OpenAI’s voice assistant, but fully local.

## 📂 Project Structure

```
├── desktop-voice-assist/
│   ├── uv.lock
│   ├── pyproject.toml
│   ├── print_structure.py
│   ├── README.md
│   ├── .gitignore
│   ├── .python-version
│   ├── benchmarks/
│   │   ├── results/
│   │   ├── scripts/
│   │   │   ├── benchmark_trancription.py
│   │   │   ├── benchmark_models .py
│   │   ├── notebooks/
│   ├── data/
│   │   ├── context.json           # Created on Startup
│   │   ├── samples/
│   │   ├── audio_input/
│   ├── src/
│   │   ├── voice_assist/
│   │   │   ├── main.py
│   │   │   ├── tools/
│   │   │   │   ├── tools.py
│   │   │   │   ├── get_active_apps.py
│   │   │   ├── llm/
│   │   │   │   ├── ai_agent.py
│   │   │   │   ├── llm_process.py
│   │   │   │   ├── llm_stream.py
│   │   │   ├── pipelines/
│   │   │   │   ├── multiprocess_pipeline.py
│   │   │   │   ├── voice_assistant_pipeline.py
│   │   │   ├── utils/
│   │   │   │   ├── context_manager.py
│   │   │   ├── transacription/
│   │   │   │   ├── transcriber.py
│   │   │   │   ├── auto_spliter.py
│   │   │   ├── voice/
│   │   │   │   ├── voice_process.py
│   │   │   │   ├── voice_synth.py
│   │   │   │   ├── voice_stream.py
│   │   │   │   ├── kokoro/                # Stores the kokoro files that the voice synthesizer uses
│   │   │   │   │   ├── kokoro-v1.0.onnx
│   │   │   │   │   ├── voices-v1.0.bin
```

## 🚀 Getting Started

## Setup

Please review the instructions below

<details>

<summary>Instructions</summary>

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation) for isolated Python (Recommend).

```console
pip install uv
```

2. Download the repo
   
3. From the repo
   
    ```console
    uv sync
    ```
   
4. Download the files [`kokoro-v1.0.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx), and [`voices-v1.0.bin`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin) and place them into the src/voice_assist/voice/kokoro

5. Run

```console
uv run main.py
```

</details>

</details>

### Prerequisites

* Python 3.11+
* [Ollama](https://ollama.ai) installed and configured

### Run the Assistant

```bash
uv sync 
uv run main.py
```

### Example Interaction

* You: "open Google Chrome."
* Assistant → JSON: `{ "type": "open_app", "name": "Google Chrome" }`

## 📊 Benchmarks

The `benchmarks` folder contains tools for testing different models. Run benchmarks with:

This will output performance metrics for speed, accuracy, and responsiveness.

## 🤝 Contributing

Contributions are welcome! Feel free to submit pull requests for:

* New features
* Bug fixes
* Model integrations
* Benchmarks improvements

## 📄 License

kokoro-onnx: MIT
kokoro model: Apache 2.0
