"""Task registry for Instinct-RL based mjlab tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from instinct_mjlab.rl import InstinctRlOnPolicyRunnerCfg
from mjlab.envs import ManagerBasedRlEnvCfg


@dataclass
class _TaskCfg:
  env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg]
  play_env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg]
  instinct_rl_cfg_factory: Callable[[], InstinctRlOnPolicyRunnerCfg]
  runner_cls: type | None


_REGISTRY: dict[str, _TaskCfg] = {}


def register_instinct_task(
  task_id: str,
  env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg],
  play_env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg],
  instinct_rl_cfg_factory: Callable[[], InstinctRlOnPolicyRunnerCfg],
  runner_cls: type | None = None,
) -> None:
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered.")
  _REGISTRY[task_id] = _TaskCfg(
    env_cfg_factory,
    play_env_cfg_factory,
    instinct_rl_cfg_factory,
    runner_cls,
  )


def list_tasks() -> list[str]:
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  task_cfg = _REGISTRY[task_name]
  return task_cfg.play_env_cfg_factory() if play else task_cfg.env_cfg_factory()


def load_instinct_rl_cfg(task_name: str) -> InstinctRlOnPolicyRunnerCfg:
  return _REGISTRY[task_name].instinct_rl_cfg_factory()


def load_runner_cls(task_name: str) -> type | None:
  return _REGISTRY[task_name].runner_cls
