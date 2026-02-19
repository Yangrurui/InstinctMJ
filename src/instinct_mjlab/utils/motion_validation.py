"""Tracking motion file validation and discovery utilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

_TRACKING_MOTION_REQUIRED_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


_DEFAULT_DATASET_CANDIDATES = (
  "~/Xyk/datasets",
  "~/Xyk/Datasets",
  "~/Datasets",
)


def resolve_datasets_root() -> Path:
  """Resolve datasets root from override or known local workspace candidates."""
  override = os.environ.get("INSTINCT_DATASETS_ROOT", "").strip()
  if override:
    return Path(override).expanduser().resolve()

  for candidate in _DEFAULT_DATASET_CANDIDATES:
    candidate_path = Path(candidate).expanduser()
    if candidate_path.exists() and candidate_path.is_dir():
      return candidate_path.resolve()

  # Keep a deterministic fallback even when no candidate exists yet.
  return Path(_DEFAULT_DATASET_CANDIDATES[1]).expanduser().resolve()


def _tracking_motion_missing_keys(motion_path: Path) -> tuple[str, ...]:
  with np.load(motion_path) as data:
    keys = set(data.files)
  missing = tuple(key for key in _TRACKING_MOTION_REQUIRED_KEYS if key not in keys)
  return missing


def validate_tracking_motion_file(motion_path: Path) -> None:
  """Validate motion npz schema expected by `mjlab.tasks.tracking.mdp.MotionLoader`."""
  if not motion_path.exists() or not motion_path.is_file():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")
  missing_keys = _tracking_motion_missing_keys(motion_path)
  if missing_keys:
    raise ValueError(
      "motion 文件格式不匹配 mjlab tracking 期望字段："
      f" {motion_path}\n"
      f"缺失 keys: {list(missing_keys)}\n"
      f"期望 keys: {list(_TRACKING_MOTION_REQUIRED_KEYS)}"
    )


def _find_parkour_motion_from_task_cfg() -> Path | None:
  """Resolve Parkour default motion from task config (InstinctLab-style path/yaml)."""
  try:
    from instinct_mjlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import (
      AmassMotionCfg,
    )
  except Exception:
    return None

  configured_root_str = str(getattr(AmassMotionCfg, "path", "")).strip()
  if not configured_root_str:
    return None
  configured_root = Path(configured_root_str).expanduser()

  yaml_path_str = getattr(AmassMotionCfg, "filtered_motion_selection_filepath", None)
  candidate_paths: list[Path] = []

  if yaml_path_str is not None and str(yaml_path_str).strip():
    yaml_path = Path(str(yaml_path_str)).expanduser()
    if yaml_path.exists() and yaml_path.is_file():
      try:
        with yaml_path.open("r", encoding="utf-8") as file:
          loaded_yaml = yaml.safe_load(file) or {}
      except (yaml.YAMLError, OSError):
        yaml_data = {}
      else:
        yaml_data = loaded_yaml if isinstance(loaded_yaml, dict) else {}
      selected_files = yaml_data.get("selected_files", [])
      if isinstance(selected_files, list):
        for selected_file in selected_files:
          selected_text = str(selected_file).strip()
          if not selected_text:
            continue
          selected_path = Path(selected_text).expanduser()
          if selected_path.is_absolute():
            candidate_paths.append(selected_path)
          else:
            candidate_paths.append(configured_root / selected_path)
            candidate_paths.extend(configured_root.rglob(selected_path.name))

  for pattern in ("**/motion.npz", "**/*parkour*motion*.npz", "**/*parkour*.npz"):
    candidate_paths.extend(configured_root.glob(pattern))

  seen: set[Path] = set()
  for path in candidate_paths:
    resolved_path = path.expanduser().resolve()
    if resolved_path in seen:
      continue
    seen.add(resolved_path)
    if not resolved_path.exists() or not resolved_path.is_file():
      continue
    try:
      validate_tracking_motion_file(resolved_path)
    except (ValueError, OSError, FileNotFoundError):
      continue
    return resolved_path
  return None


def find_default_tracking_motion_file(task_id: str) -> Path | None:
  """Best-effort search for a usable tracking motion file from datasets root."""
  if "Parkour" in task_id:
    from_task_cfg = _find_parkour_motion_from_task_cfg()
    if from_task_cfg is not None:
      return from_task_cfg

  datasets_root = resolve_datasets_root()
  if not datasets_root.exists() or not datasets_root.is_dir():
    return None

  candidate_patterns: list[str] = ["**/motion.npz"]
  if "Parkour" in task_id:
    candidate_patterns = [
      "**/*parkour*motion*.npz",
      "**/*parkour*.npz",
      *candidate_patterns,
    ]

  for pattern in candidate_patterns:
    for path in sorted(datasets_root.glob(pattern)):
      if not path.is_file():
        continue
      try:
        validate_tracking_motion_file(path)
      except (ValueError, OSError, FileNotFoundError):
        continue
      return path.resolve()
  return None
