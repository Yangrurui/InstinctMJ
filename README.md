# Instinct_mjlab

`Instinct_mjlab` ports InstinctLab tasks to `mjlab` and runs training/playback with `instinct_rl`.

## Reference to Original InstinctLab

- InstinctLab repo: `https://github.com/project-instinct/instinctlab`
- Original root README: `https://github.com/project-instinct/instinctlab/blob/main/README.md`
- Original tasks root: `source/instinctlab/instinctlab/tasks`
- Local mirror in this workspace: `../InstinctLab`
- Local tasks root in this workspace: `../InstinctLab/source/instinctlab/instinctlab/tasks`

## Prerequisites

- Python `>=3.10,<3.14`
- Linux + NVIDIA GPU (current training pipeline requires CUDA)
- Source checkouts of both:
  - `mjlab`
  - `instinct_rl`

## Installation

Clone required source repositories as sibling directories:

```bash
mkdir -p ~/Project-Instinct
cd ~/Project-Instinct

git clone https://github.com/mujocolab/mjlab.git
git clone https://github.com/project-instinct/instinct_rl.git
# If you do not already have this repository locally:
git clone <your-instinct-mjlab-repo-url> Instinct_mjlab
```

Then install from `Instinct_mjlab`:

```bash
cd Instinct_mjlab
uv sync
```

If you prefer `pip`:

```bash
pip install -e ../mjlab
pip install -e ../instinct_rl
pip install -e .
```

## List Available Tasks

```bash
instinct-list-envs
instinct-list-envs shadowing
```

Registered task IDs:

- `Instinct-Locomotion-Flat-G1-v0`
- `Instinct-Locomotion-Flat-G1-Play-v0`
- `Instinct-BeyondMimic-Plane-G1-v0`
- `Instinct-BeyondMimic-Plane-G1-Play-v0`
- `Instinct-Shadowing-WholeBody-Plane-G1-v0`
- `Instinct-Shadowing-WholeBody-Plane-G1-Play-v0`
- `Instinct-Perceptive-Shadowing-G1-v0`
- `Instinct-Perceptive-Shadowing-G1-Play-v0`
- `Instinct-Perceptive-Vae-G1-v0`
- `Instinct-Perceptive-Vae-G1-Play-v0`
- `Instinct-Parkour-Target-Amp-G1-v0`
- `Instinct-Parkour-Target-Amp-G1-Play-v0`

## Train and Play

Train:

```bash
instinct-train Instinct-Locomotion-Flat-G1-v0
instinct-train Instinct-Perceptive-Shadowing-G1-v0
```

Play (`--load-run` is required):

```bash
instinct-play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
instinct-play Instinct-Perceptive-Shadowing-G1-Play-v0 --load-run <run_name>
```

Export ONNX (Parkour):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

Module form (if console scripts are not available):

```bash
python -m instinct_mjlab.scripts.instinct_rl.train Instinct-Locomotion-Flat-G1-v0
python -m instinct_mjlab.scripts.instinct_rl.play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
python -m instinct_mjlab.scripts.list_envs
```

## Data and Logs

- Override dataset root with `INSTINCT_DATASETS_ROOT` when needed.
- Training logs: `logs/instinct_rl/<experiment_name>/<timestamp_run>/`
- Play videos: under `videos/play/` in the selected run directory.

## Task Documentation

- Shadowing: `src/instinct_mjlab/tasks/shadowing/README.md`
- Parkour: `src/instinct_mjlab/tasks/parkour/README.md`
