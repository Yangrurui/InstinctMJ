# Shadowing Task

## InstinctLab Original Sources

- InstinctLab repository: `https://github.com/project-instinct/instinctlab`
- Original shadowing README:
  `https://github.com/project-instinct/instinctlab/blob/main/source/instinctlab/instinctlab/tasks/shadowing/README.md`
- Original perceptive shadowing config:
  `source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py`
- Local shadowing README in this workspace:
  `../InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/README.md`
- Local perceptive shadowing config in this workspace:
  `../InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py`

## Prerequisite

Install `mjlab` and `instinct_rl` source code first (see `Instinct_mjlab/README.md` for full setup), then install this package so `instinct-train` and `instinct-play` are available.

## Basic Usage Guidelines

### BeyondMimic Shadowing

**Task IDs:**
- `Instinct-BeyondMimic-Plane-G1-v0` (train)
- `Instinct-BeyondMimic-Plane-G1-Play-v0` (play)

This is an exact replication of the BeyondMimic training configuration.

1. Go to `beyondmimic/config/g1/beyondmimic_plane_cfg.py` and set the motion source:

    - `MOTION_NAME`: An identifier for the motion setup you are using.
    - `AmassMotionCfg.path`: The folder path to where you store the motion files.
    - `_hacked_selected_file_`: The filename of the motion you want to use, relative to the `AmassMotionCfg.path` folder.

    You can also use environment variables:
    - `INSTINCT_BEYONDMIMIC_MOTION_PATH`
    - `INSTINCT_BEYONDMIMIC_SELECTED_FILE`

2. Train the policy:
```bash
instinct-train Instinct-BeyondMimic-Plane-G1-v0
```

3. Play trained policy (`--load-run` is required; absolute path is recommended, or use `--no-resume` for untrained policy):
```bash
instinct-play Instinct-BeyondMimic-Plane-G1-Play-v0 --load-run <run_name>
```

### Whole Body Shadowing

**Task IDs:**
- `Instinct-Shadowing-WholeBody-Plane-G1-v0` (train)
- `Instinct-Shadowing-WholeBody-Plane-G1-Play-v0` (play)

1. Go to `whole_body/config/g1/plane_shadowing_cfg.py` and set the `MOTION_NAME`, `_path_`, `_hacked_selected_files_` to the motion you want to use.

    - `MOTION_NAME`: An identifier for the motion setup you are using.
    - `_path_`: The folder path to where you store the motion files.
    - `_hacked_selected_files_`: The filenames of the motion you want to use, relative to the `_path_` folder.
    - Optional env var: `INSTINCT_WHOLEBODY_MOTION_PATH`

2. Train the policy:
```bash
instinct-train Instinct-Shadowing-WholeBody-Plane-G1-v0
```

3. Play trained policy (`--load-run` is required; absolute path is recommended, or use `--no-resume` for untrained policy):
```bash
instinct-play Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --load-run <run_name>
```

### Perceptive Shadowing

**Task IDs:**
- `Instinct-Perceptive-Shadowing-G1-v0` (train)
- `Instinct-Perceptive-Shadowing-G1-Play-v0` (play)

1. Configure perceptive motion/terrain before training or play.

    Required:
    - `INSTINCT_PERCEPTIVE_MOTION_FOLDER`: motion dataset folder. The task reads `metadata.yaml` from this folder.

    Optional:
    - `INSTINCT_PERCEPTIVE_FORCE_PLANE`: force training terrain to plane (`true`/`false`, default: `false`)
    - `INSTINCT_PERCEPTIVE_PLAY_MOTION_PATH`: play-only motion folder override
    - `INSTINCT_PERCEPTIVE_PLAY_STUB_SAMPLING_STRATEGY`: play stub sampling strategy (default: `independent`)
    - `INSTINCT_PERCEPTIVE_PLAY_MOTION_BIN_LENGTH_S`: play motion bin length (`auto`/`none`/float, default: `auto`)
    - `INSTINCT_PERCEPTIVE_PLAY_FORCE_PLANE`: force play terrain to plane (`true`/`false`, default: `false`)

2. Train the policy:
```bash
instinct-train Instinct-Perceptive-Shadowing-G1-v0
```

3. Play trained policy (`--load-run` is required; absolute path is recommended, or use `--no-resume` for untrained policy):
```bash
instinct-play Instinct-Perceptive-Shadowing-G1-Play-v0 --load-run <run_name>
```

## Common Options

- `--num-envs`: Number of parallel environments (default varies by task)
- `--load-run`: Run name/path pattern to select a checkpoint for play
- `--device`: Runtime device, e.g. `cuda:0`
- `--viewer`: Viewer backend (`none`/`native` for train, `auto`/`native`/`viser`/`none` for play)
- `--video`: Record training/playback videos

Module form (if console scripts are not available):

```bash
python -m instinct_mjlab.scripts.instinct_rl.train Instinct-Perceptive-Shadowing-G1-v0
python -m instinct_mjlab.scripts.instinct_rl.play Instinct-Perceptive-Shadowing-G1-Play-v0 --load-run <run_name>
```
