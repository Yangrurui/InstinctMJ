import copy
from dataclasses import MISSING, dataclass, field
from typing import List
import uuid

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    FlatPatchSamplingCfg,
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)
from mjlab.terrains.utils import find_flat_patches_from_heightfield


def _unwrap_height_field_function(function_obj: object) -> object:
    """Unwrap decorator layers and return the raw `(difficulty, cfg) -> heights` callable."""
    raw_function = function_obj
    while hasattr(raw_function, "__wrapped__"):
        raw_function = raw_function.__wrapped__
    return raw_function


def _add_wall_geometries(
    *,
    body: mujoco.MjsBody,
    cfg: "HfTerrainBaseCfg",
    rng: np.random.Generator,
) -> list[TerrainGeometry]:
    """Add optional side walls, matching legacy `generate_wall` semantics."""
    if not hasattr(cfg, "wall_prob"):
        return []

    wall_prob = getattr(cfg, "wall_prob")
    if wall_prob is None:
        return []

    wall_height = float(getattr(cfg, "wall_height", 5.0))
    wall_thickness = float(getattr(cfg, "wall_thickness", 0.05))
    if wall_height <= 0.0 or wall_thickness <= 0.0:
        return []

    min_x, max_x = 0.0, float(cfg.size[0])
    min_y, max_y = 0.0, float(cfg.size[1])
    geoms: list[TerrainGeometry] = []

    # Left wall
    if rng.uniform() < wall_prob[0]:
        left_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(wall_thickness * 0.5, (max_y - min_y) * 0.5, wall_height * 0.5),
            pos=(min_x - wall_thickness * 0.5, (min_y + max_y) * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=left_wall))

    # Right wall
    if rng.uniform() < wall_prob[1]:
        right_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(wall_thickness * 0.5, (max_y - min_y) * 0.5, wall_height * 0.5),
            pos=(max_x + wall_thickness * 0.5, (min_y + max_y) * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=right_wall))

    # Front wall
    if rng.uniform() < wall_prob[2]:
        front_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=((max_x - min_x) * 0.5, wall_thickness * 0.5, wall_height * 0.5),
            pos=((min_x + max_x) * 0.5, min_y - wall_thickness * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=front_wall))

    # Back wall
    if rng.uniform() < wall_prob[3]:
        back_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=((max_x - min_x) * 0.5, wall_thickness * 0.5, wall_height * 0.5),
            pos=((min_x + max_x) * 0.5, max_y + wall_thickness * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=back_wall))

    return geoms


def _height_field_to_output(
    *,
    heights: np.ndarray,
    cfg: "HfTerrainBaseCfg",
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
) -> TerrainOutput:
    """Convert integer height field to MuJoCo hfield + TerrainOutput."""
    body = spec.body("terrain")
    heights_i16 = np.asarray(np.rint(heights), dtype=np.int16)

    elevation_min = int(np.min(heights_i16))
    elevation_max = int(np.max(heights_i16))
    elevation_range_i16 = elevation_max - elevation_min
    elevation_range_i16 = elevation_range_i16 if elevation_range_i16 > 0 else 1

    normalized_elevation = (heights_i16 - elevation_min) / elevation_range_i16
    max_physical_height = float(elevation_range_i16) * float(cfg.vertical_scale)
    base_thickness = max_physical_height * float(getattr(cfg, "base_thickness_ratio", 1.0))
    # MuJoCo hfield top height uses `geom_z + normalized * elevation_range`.
    # `base_thickness` only controls underside thickness and must NOT be
    # subtracted from geom z, otherwise each tile gets an extra depth-dependent
    # downward shift and seam steps appear between neighboring terrains.
    geom_z = float(elevation_min) * float(cfg.vertical_scale)

    unique_id = uuid.uuid4().hex
    hfield = spec.add_hfield(
        name=f"hfield_{unique_id}",
        size=[
            float(cfg.size[0]) / 2.0,
            float(cfg.size[1]) / 2.0,
            max_physical_height,
            base_thickness,
        ],
        nrow=heights_i16.shape[0],
        ncol=heights_i16.shape[1],
        userdata=normalized_elevation.astype(np.float32).flatten().tolist(),
    )

    hfield_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield.name,
        pos=(float(cfg.size[0]) * 0.5, float(cfg.size[1]) * 0.5, geom_z),
    )

    x1 = int((float(cfg.size[0]) * 0.5 - 1.0) / float(cfg.horizontal_scale))
    x2 = int((float(cfg.size[0]) * 0.5 + 1.0) / float(cfg.horizontal_scale))
    y1 = int((float(cfg.size[1]) * 0.5 - 1.0) / float(cfg.horizontal_scale))
    y2 = int((float(cfg.size[1]) * 0.5 + 1.0) / float(cfg.horizontal_scale))
    x1 = max(0, min(x1, heights_i16.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, heights_i16.shape[0]))
    y1 = max(0, min(y1, heights_i16.shape[1] - 1))
    y2 = max(y1 + 1, min(y2, heights_i16.shape[1]))
    origin_z = float(np.max(heights_i16[x1:x2, y1:y2])) * float(cfg.vertical_scale)
    origin = np.array([float(cfg.size[0]) * 0.5, float(cfg.size[1]) * 0.5, origin_z], dtype=np.float64)

    flat_patches: dict[str, np.ndarray] | None = None
    if cfg.flat_patch_sampling is not None:
        heights_phys = (heights_i16.astype(np.float64) - float(elevation_min)) * float(cfg.vertical_scale)
        z_offset = float(elevation_min) * float(cfg.vertical_scale)
        flat_patches = {}
        for patch_name, patch_cfg in cfg.flat_patch_sampling.items():
            flat_patches[patch_name] = find_flat_patches_from_heightfield(
                heights=heights_phys,
                horizontal_scale=float(cfg.horizontal_scale),
                z_offset=z_offset,
                cfg=patch_cfg,
                rng=rng,
            )

    geometries = [TerrainGeometry(geom=hfield_geom, hfield=hfield)]
    geometries.extend(_add_wall_geometries(body=body, cfg=cfg, rng=rng))
    return TerrainOutput(origin=origin, geometries=geometries, flat_patches=flat_patches)


@dataclass(kw_only=True)
class HfTerrainBaseCfg(SubTerrainCfg):
    legacy_function: object | None = None
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    base_thickness_ratio: float = 1.0

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        if self.legacy_function is None:
            raise ValueError(f"{type(self).__name__}.legacy_function is not set.")

        if self.border_width > 0.0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"The border width ({self.border_width}) must be greater than or equal to the"
                f" horizontal scale ({self.horizontal_scale})."
            )

        # NOTE:
        # MuJoCo hfield sampling in mjlab assumes grid counts use
        # `int(size / horizontal_scale)` (no `+1`), and border uses
        # `int(border_width / horizontal_scale)`.
        # Keeping Isaac-style `+1` here introduces seam artifacts between
        # neighboring tiles (boundary samples fall into mismatched cells),
        # which amplifies with deeper terrains.
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)
        border_pixels = int(self.border_width / self.horizontal_scale)
        heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        sub_terrain_size = [width_pixels - 2 * border_pixels, length_pixels - 2 * border_pixels]
        sub_terrain_size = [dim * self.horizontal_scale for dim in sub_terrain_size]

        cfg_for_gen = copy.deepcopy(self)
        cfg_for_gen.size = tuple(sub_terrain_size)
        raw_function = _unwrap_height_field_function(self.legacy_function)
        generated = raw_function(difficulty, cfg_for_gen)
        heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = np.asarray(
            np.rint(generated), dtype=np.int16
        )

        return _height_field_to_output(heights=heights, cfg=self, spec=spec, rng=rng)


@dataclass(kw_only=True)
class HfPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@dataclass(kw_only=True)
class HfInvertedPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    inverted: bool = True


@dataclass(kw_only=True)
class HfPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@dataclass(kw_only=True)
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    inverted: bool = True


@dataclass(kw_only=True)
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = MISSING
    obstacle_height_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0


@dataclass(kw_only=True)
class HfWaveTerrainCfg(HfTerrainBaseCfg):
    amplitude_range: tuple[float, float] = MISSING
    num_waves: int = 1


@dataclass(kw_only=True)
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth: float = -10.0
    platform_width: float = 1.0

from . import hf_terrains

@dataclass
class WallTerrainCfgMixin:
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    wall_prob: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls

@dataclass(kw_only=True)
class PerlinPlaneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    legacy_function: object = hf_terrains.perlin_plane_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    noise_scale: float | List[float] = 0.05
    noise_frequency: int = 20

    fractal_octaves: int = 2

    fractal_lacunarity: float = 2.0

    fractal_gain: float = 0.25

    centering: bool = False  # If True, the noise will be centered around 0

@dataclass(kw_only=True)
class PerlinPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    legacy_function: object = hf_terrains.perlin_pyramid_sloped_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinInvertedPyramidSlopedTerrainCfg(HfInvertedPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    legacy_function: object = hf_terrains.perlin_pyramid_sloped_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = True
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    legacy_function: object = hf_terrains.perlin_pyramid_stairs_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinInvertedPyramidStairsTerrainCfg(HfInvertedPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    legacy_function: object = hf_terrains.perlin_pyramid_stairs_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = True
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg, WallTerrainCfgMixin):
    legacy_function: object = hf_terrains.perlin_discrete_obstacles_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = MISSING
    obstacle_height_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinWaveTerrainCfg(HfWaveTerrainCfg, WallTerrainCfgMixin):
    legacy_function: object = hf_terrains.perlin_wave_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    amplitude_range: tuple[float, float] = MISSING
    num_waves: int = 1
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinSteppingStonesTerrainCfg(HfSteppingStonesTerrainCfg, WallTerrainCfgMixin):
    legacy_function: object = hf_terrains.perlin_stepping_stones_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth: float = -10.0
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

# -- Newly added terrain configurations for parkour terrains-- #
@dataclass(kw_only=True)
class PerlinParapetTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a parapet terrain, can be used for jump and hurdle tasks."""

    legacy_function: object = hf_terrains.perlin_parapet_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    parapet_height: tuple[float, float] | float = (0.1, 0.3)
    parapet_length: tuple[float, float] | float = (0.1, 0.3)
    parapet_width: float | None = None
    curved_top_rate: float | None = None
    """The rate to generate curved top. If None, the top will be flat."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinGutterTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a gutter parkour terrain."""

    legacy_function: object = hf_terrains.perlin_gutter_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    gutter_length: tuple[float, float] | float = (0.5, 1.5)  # the distance between gutters
    gutter_depth: tuple[float, float] | float = (0.1, 0.3)  # the depth of the gutter
    gutter_width: float | None = None  # the length of the gutter
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinStairsUpDownTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs up and down parkour terrain."""

    legacy_function: object = hf_terrains.perlin_stairs_up_down_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinStairsDownUpTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs down and up parkour terrain."""

    legacy_function: object = hf_terrains.perlin_stairs_down_up_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinTiltTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilt terrain."""

    legacy_function: object = hf_terrains.perlin_tilt_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    wall_height: tuple[float, float] | float = MISSING
    wall_width: float | None = None
    wall_length: tuple[float, float] | float = MISSING
    wall_opening_angle: tuple[float, float] | float = MISSING  # in degrees
    wall_opening_width: tuple[float, float] | float = MISSING
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinTiltedRampTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilted ramp terrain."""

    legacy_function: object = hf_terrains.perlin_tilted_ramp_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    tilt_angle: tuple[float, float] | float = MISSING  # in degrees
    tilt_height: tuple[float, float] | float = MISSING
    tilt_width: tuple[float, float] | float = MISSING
    tilt_length: tuple[float, float] | float = MISSING
    switch_spacing: tuple[float, float] | float = MISSING
    spacing_curriculum: bool | None = None
    overlap_size: float | None = None
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinSlopeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a slope up and down terrain with a flat ground in the middle."""

    legacy_function: object = hf_terrains.perlin_slope_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_angle: tuple[float, float] | float = MISSING  # in degrees
    per_slope_length: tuple[float, float] | float = MISSING
    platform_length: float = 1.0
    slope_width: float | None = None
    up_down: bool | None = None  # If True or None, the slope will be up and down, otherwise it will be down and up.
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinCrossStoneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a cross stone terrain."""

    legacy_function: object = hf_terrains.perlin_cross_stone_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    stone_size: tuple[float, float] = MISSING
    stone_height: tuple[float, float] | float = MISSING
    stone_spacing: tuple[float, float] | float = MISSING
    ground_depth: float = -0.5
    platform_width: float = 1.5
    xy_random_ratio: float = 0.2
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

@dataclass(kw_only=True)
class PerlinSquareGapTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    legacy_function: object = hf_terrains.perlin_square_gap_terrain
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    gap_distance_range: tuple[float, float] = (0.1, 0.5)
    gap_depth: tuple[float, float] = (0.2, 0.5)
    platform_width: float = 1.5
    border_width: float = 0.0

    perlin_cfg: PerlinPlaneTerrainCfg | None = None
