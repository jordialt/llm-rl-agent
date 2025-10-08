# 🧠 LLM + RL Agent (CartPole-v1)

This repository demonstrates a small experiment combining Reinforcement Learning (RL) and a lightweight Large Language Model (LLM). The RL side uses PPO (from Stable-Baselines3) to learn the `CartPole-v1` task, while a FLAN-T5 LLM provides short natural-language commentary and advice about the agent's state and performance.

The goal is educational: show how LLM-driven observations or hints can be integrated into an RL training loop in a compact, CPU-friendly demo.

## Highlights

- PPO agent (Stable-Baselines3)
- Lightweight LLM feedback using Hugging Face `transformers` and `google/flan-t5-small`
- Designed to run on CPU
- Minimal, easy-to-read code to extend for other environments or LLM prompts

## Quick overview

- Training script: `src/train_agent.py` — runs a small PPO training loop and requests one-line advice from the LLM after each episode.
- LLM helper: `src/llm_helper.py` — wraps a FLAN-T5 seq2seq model (tokenizer + model) and exposes a simple `generate(prompt)` method.

## Requirements

The project uses Python 3.10+. Dependencies are listed in `requirements.txt` (the file may be empty in the repo snapshot — install the packages below if needed):

- gymnasium
- stable-baselines3
- torch (CPU build is fine)
- transformers

You can install these with pip. On a Linux machine with bash:

```bash
python -m pip install --upgrade pip
pip install gymnasium stable-baselines3 torch transformers
```

If you use a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install gymnasium stable-baselines3 torch transformers
```

Notes:

- `torch` will install the CPU version by default on many systems. If you have CUDA and want GPU support, install the appropriate `torch` build from https://pytorch.org.
- If `requirements.txt` is later populated, prefer `pip install -r requirements.txt`.

## Usage — Quick start

Run the training demo (keeps training small for a fast demo):

```bash
python src/train_agent.py
```

What the script does:

- Creates a `CartPole-v1` environment
- Instantiates a PPO agent (Stable-Baselines3)
- Loads the FLAN-T5 small model via `src/llm_helper.py`
- Runs a few episodes; after each episode it asks the LLM for one line of advice based on the final observation
- Continues learning and saves the final model to `ppo_cartpole_llm` in the working directory

Expected output (truncated):

- Printed messages about the LLM model loading
- Episode reward summaries and a one-line LLM advice per episode
- A saved model file `ppo_cartpole_llm.zip` (Stable-Baselines3 saves model data)

## Files and purpose

- `src/train_agent.py` — main script demonstrating the training loop with LLM commentary. Keep in mind it's a minimal demo (short training timesteps for speed).
- `src/llm_helper.py` — simple wrapper around Hugging Face `AutoTokenizer` and `AutoModelForSeq2SeqLM` for the `google/flan-t5-small` model. Exposes `generate(prompt)`.
- `requirements.txt` — recommended dependencies (may need to be populated).
- `README.md` — this document.

## Development notes & extension ideas

This demo is intentionally small and designed for experimentation. Here are ideas to extend it:

- Replace FLAN-T5 with a different model or a hosted LLM (e.g., OpenAI) and compare quality / latency.
- Use the LLM to produce reward-shaping hints or to propose subgoals rather than only one-line advice.
- Collect LLM responses and analyze whether advice correlates with improved learning.
- Add command-line arguments to `train_agent.py` to configure timesteps, model names, and LLM prompts.
- Add unit tests for `llm_helper.py` (mock the model) and for `train_agent.py` helper functions.

## Troubleshooting

- Transformers model download: the first run will download the FLAN-T5 weights (tens to hundreds of MB). Ensure you have internet access and sufficient disk space.
- If you see CUDA-related errors but you intended CPU-only, ensure `torch` was installed without CUDA or set the device to `cpu` when creating the SB3 model (the demo already sets `device="cpu"`).
- If `gymnasium` fails to create `CartPole-v1`, try installing `gym` instead or check the gymnasium version compatibility.

## Reproducibility and testing

The code is minimal and not configured for strict reproducibility (no fixed random seeds). To reproduce runs exactly, set seeds in the environment, numpy, torch, and Stable-Baselines3 when instantiating the model.

## Contact

Questions or collaboration ideas? Open an issue or contact the repository owner.

---


