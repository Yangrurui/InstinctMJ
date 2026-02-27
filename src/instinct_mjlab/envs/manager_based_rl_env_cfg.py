from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.viewer.viewer_config import ViewerConfig


@dataclass
class InstinctLabRLEnvCfg(ManagerBasedRlEnvCfg):
  """Configuration for a reinforcement learning environment with the manager-based workflow."""

  viewer: ViewerConfig = field(default_factory=ViewerConfig)
  """Viewer Settings."""

  # monitor settings
  monitors: object | None = None
  """Monitor Settings.

  Please refer to the `instinct_mjlab.monitors.MonitorManager` class for more details.
  """
