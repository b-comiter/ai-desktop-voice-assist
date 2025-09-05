import time
import statistics
import csv
import json
import platform
import psutil
from pathlib import Path
from ollama import Client
from colorama import Fore, init

client = Client()

# List of base models (without specifying quant tag)
MODELS = [
    "mistral-nemo:12b",
    "llama3",
    "llama3.1:8b",
    "qwen3:0.6b",
    "qwen2.5:1.5b-instruct",
    "gemma2:2b",
    "tinyllama",
    "llama3.2:1b",
    "llama3.2:3b",
    "phi3:mini",
    "phi4-mini:3.8b",
]

PROMPT = (
    "Summarize the benefits of using small language models for local CPU inference."
)
TRIALS = 3  # number of times to run each model

OUTPUT_DIR = Path("benchmarks/results")
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_FILE = OUTPUT_DIR / "model_benchmarks.csv"
JSON_FILE = OUTPUT_DIR / "model_benchmarks.json"


def get_system_info():
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }


def is_model_installed(model_name):
    """
    Checks if a given model is installed in Ollama.

    Args:
        model_name (str): The name of the model to check (e.g., "llama3").

    Returns:
        bool: True if the model is installed, False otherwise.
    """
    try:
        models = client.list()
        for model_info in models["models"]:
            if model_info["model"] == model_name:
                return True
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def ensure_model(model_name: str):
    """Pull model if not present locally, automatically picking CPU-friendly quant."""
    if not is_model_installed(model_name):
        print(f"Model '{model_name}' not found locally. Pulling...")
        client.pull(model_name)
        print(f"Model '{model_name}' pulled successfully.")

    return model_name


def benchmark_model(model: str, trials: int = 3):
    times, toks_per_sec, outputs = [], [], []

    for i in range(trials):
        start = time.time()
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            options={
                "num_thread": 0,  # auto-detect threads
                "num_ctx": 2048,  # keep modest for CPU speed
                "temperature": 0,  # deterministic
            },
        )
        end = time.time()

        content = response["message"]["content"]
        total_time = end - start
        token_count = response.get("eval_count", len(content.split()))
        speed = token_count / total_time if total_time > 0 else 0

        times.append(total_time)
        toks_per_sec.append(speed)
        outputs.append(content)

    return {
        "model": model,
        "avg_time": round(statistics.mean(times), 2),
        "std_time": round(statistics.pstdev(times), 2),
        "avg_speed": round(statistics.mean(toks_per_sec), 2),
        "std_speed": round(statistics.pstdev(toks_per_sec), 2),
        "output_preview": outputs[0][:200] + ("..." if len(outputs[0]) > 200 else ""),
    }


def save_results_csv(results, file_path: Path, sys_info: dict):
    with open(file_path, "w", newline="") as f:
        fieldnames = [
            "model",
            "avg_time",
            "std_time",
            "avg_speed",
            "std_speed",
            "output_preview",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    sys_file = file_path.with_name("system_info.csv")
    with open(sys_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sys_info.keys())
        writer.writeheader()
        writer.writerow(sys_info)


def save_results_json(results, file_path: Path, sys_info: dict):
    with open(file_path, "w") as f:
        json.dump({"system_info": sys_info, "benchmarks": results}, f, indent=2)


if __name__ == "__main__":
    sys_info = get_system_info()
    print("=== System Info ===")
    for k, v in sys_info.items():
        print(f"{k}: {v}")

    results = []
    for m in MODELS:
        fast_model = ensure_model(m)  # auto-pull + CPU-friendly quant
        print(f"\n=== Benchmarking {fast_model} ===")
        res = benchmark_model(fast_model, TRIALS)
        results.append(res)
        print(
            f"Avg Time: {res['avg_time']}s ±{res['std_time']} | "
            f"Speed: {res['avg_speed']} tok/s ±{res['std_speed']}"
        )
        print(f"Output Preview: {res['output_preview']}")

    # Save results
    save_results_csv(results, CSV_FILE, sys_info)
    save_results_json(results, JSON_FILE, sys_info)

    print(f"\n=== Summary saved to {OUTPUT_DIR}/ ===")
    for r in results:
        print(
            f"{r['model']}: {r['avg_time']}s ±{r['std_time']}, "
            f"{r['avg_speed']} tok/s ±{r['std_speed']}"
        )
