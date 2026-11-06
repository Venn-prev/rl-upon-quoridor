# Qudor AI

A local browser game for Quoridor, with one Gumbel AlphaZero training pipeline and the
strongest published checkpoint, `gen85_best.pt`.

## Play

Use Python 3.11 and install the CUDA-enabled environment as described below, then run:

```powershell
& "$env:USERPROFILE\venvs\qudor-ai\Scripts\python.exe" .\app.py
```

Open <http://127.0.0.1:8765>. The viewer loads `checkpoints/gen85_best.pt` by default.

## Train on an RTX 3060 (6 GB)

The project is configured for Gumbel AlphaZero with the champion's 128-channel, 10-block
network. Self-play uses one worker to avoid allocating a separate CUDA context per process;
mixed-precision training, a 128-position batch, and a disk-backed replay buffer keep the
run within a 6 GB graphics card. Replay and checkpoints are written to
`runs/rtx3060_6gb` on the D: drive, inside this repository; the run directory is ignored
by Git.

In PowerShell:

```powershell
py -3.11 -m venv "$env:USERPROFILE\venvs\qudor-ai"
& "$env:USERPROFILE\venvs\qudor-ai\Scripts\python.exe" -m pip install -r .\requirements.txt
& "$env:USERPROFILE\venvs\qudor-ai\Scripts\python.exe" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU')"
& "$env:USERPROFILE\venvs\qudor-ai\Scripts\python.exe" .\train.py --dry-run
& "$env:USERPROFILE\venvs\qudor-ai\Scripts\python.exe" .\train.py
```

The CUDA wheel includes its CUDA runtime; install a current NVIDIA driver, but do not
install a separate CUDA Toolkit just for PyTorch. The first run starts from the weights in
`gen85_best.pt` with a fresh optimiser and replay buffer. Later runs resume from
`runs/rtx3060_6gb/latest.pt`.

To select another output directory or checkpoint:

```powershell
& "$env:USERPROFILE\venvs\qudor-ai\Scripts\python.exe" .\train.py --output "C:\qudor-runs\main"
& "$env:USERPROFILE\venvs\qudor-ai\Scripts\python.exe" .\train.py --init "C:\path\to\weights.pt" --output "C:\qudor-runs\new"
```

`--force-cpu` is available for a smoke run, but training on CPU is not recommended.
The lightweight test config is `configs/az_smoke.json`.

To play a newly promoted checkpoint in the browser, copy the run's `best.pt` to
`checkpoints/rtx3060_best.pt`; the viewer scans checkpoint files in the repository.

## Kept components

- `app.py`, `index.html`, and `english.js`: local browser game.
- `quoridor_ai/core/`: Quoridor rules and the versioned state encoder.
- `quoridor_ai/az_selfplay.py`, `az_train.py`, and `az_arena.py`: Gumbel AlphaZero
  self-play, training, and promotion gating.
- `checkpoints/gen85_best.pt`: strongest published official-rules model.
- `configs/rtx3060_6gb.json`: local 6 GB GPU profile.

Run the retained tests with:

```powershell
& "$env:USERPROFILE\venvs\qudor-ai\Scripts\python.exe" -m pytest -m "not integration" -q
```

See [MODEL_CARD.md](./MODEL_CARD.md) for the champion checkpoint and its evaluation
evidence.
