from dataclasses import MISSING, dataclass, field
from typing import List
import uuid

import mujoco
import numpy as np
import trimesh

from mjlab.terrains import SubTerrainCfg as SubTerrainBaseCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from ..height_field.hf_terrains_cfg import PerlinPlaneTerrainCfg
from . import mesh_terrains


def _normalize_legacy_meshes(
    meshes: trimesh.Trimesh | list[trimesh.Trimesh] | tuple[trimesh.Trimesh, ...],
) -> list[trimesh.Trimesh]:
    if isinstance(meshes, trimesh.Trimesh):
        meshes_list = [meshes]
    elif isinstance(meshes, (list, tuple)):
        meshes_list = list(meshes)
    else:
        raise TypeError(
            "Legacy terrain function must return trimesh.Trimesh or a list/tuple of trimesh.Trimesh."
        )
    for mesh in meshes_list:
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Legacy terrain function returned a non-trimesh mesh entry.")
    return meshes_list


def _sample_legacy_mesh_flat_patches(
    *,
    meshes_list: list[trimesh.Trimesh],
    origin: np.ndarray,
    cfg: SubTerrainBaseCfg,
) -> dict[str, np.ndarray] | None:
    patch_sampling = getattr(cfg, "flat_patch_sampling", None)
    if patch_sampling is None:
        return None

    from instinct_mjlab.terrains.terrain_generator import _find_flat_patches_on_surface_mesh

    if len(meshes_list) == 0:
        raise RuntimeError("Legacy terrain function returned an empty mesh list.")
    if len(meshes_list) == 1:
        sampling_mesh = meshes_list[0]
    else:
        sampling_mesh = trimesh.util.concatenate(meshes_list)

    origin = np.asarray(origin, dtype=np.float64)
    flat_patches: dict[str, np.ndarray] = {}
    for patch_name, patch_cfg in patch_sampling.items():
        sampled_local = _find_flat_patches_on_surface_mesh(
            sampling_mesh,
            device="cpu",
            num_patches=patch_cfg.num_patches,
            patch_radius=patch_cfg.patch_radius,
            origin=origin,
            x_range=patch_cfg.x_range,
            y_range=patch_cfg.y_range,
            z_range=patch_cfg.z_range,
            max_height_diff=patch_cfg.max_height_diff,
        )
        # `_find_flat_patches_on_surface_mesh` returns samples relative to `origin`.
        flat_patches[patch_name] = sampled_local + origin
    return flat_patches


def _legacy_mesh_to_terrain_output(
    *,
    meshes: trimesh.Trimesh | list[trimesh.Trimesh] | tuple[trimesh.Trimesh, ...],
    origin: np.ndarray,
    spec: mujoco.MjSpec,
    flat_patches: dict[str, np.ndarray] | None,
) -> TerrainOutput:
    """Convert legacy `(meshes, origin)` output into mjlab `TerrainOutput`."""
    meshes_list = _normalize_legacy_meshes(meshes)

    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    for mesh in meshes_list:
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        extents = bounds[1] - bounds[0]
        collapsed_axes = np.where(extents < 1.0e-9)[0]
        if len(collapsed_axes) == 1:
            # MuJoCo mesh compilation requires non-zero 3D volume.
            # Legacy terrains may include perfectly planar meshes (for example
            # random_multi_box ground with z=0). Convert them to a thin box
            # geom while preserving the original supporting plane location.
            collapsed_axis = int(collapsed_axes[0])
            thin = 1.0e-3
            center = (bounds[0] + bounds[1]) * 0.5
            center[collapsed_axis] = bounds[1, collapsed_axis] - thin * 0.5
            box_extents = extents.copy()
            box_extents[collapsed_axis] = thin
            box_geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(box_extents * 0.5).tolist(),
                pos=center.tolist(),
            )
            geometries.append(TerrainGeometry(geom=box_geom))
            continue

        mesh_name = f"legacy_mesh_{uuid.uuid4().hex}"
        spec.add_mesh(
            name=mesh_name,
            uservert=np.asarray(mesh.vertices, dtype=np.float32).reshape(-1).tolist(),
            userface=np.asarray(mesh.faces, dtype=np.int32).reshape(-1).tolist(),
        )
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name, pos=(0.0, 0.0, 0.0))
        geometries.append(TerrainGeometry(geom=geom))

    return TerrainOutput(
        origin=np.asarray(origin, dtype=np.float64),
        geometries=geometries,
        flat_patches=flat_patches,
    )

@dataclass
class WallTerrainCfgMixin:
    wall_prob: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls

@dataclass(kw_only=True)
class STLHeightfieldTerrainCfg(SubTerrainBaseCfg):
    """Configuration for STL-to-heightfield terrain generation.

    This terrain type loads an STL file and converts it to a MuJoCo heightfield
    for both visualization and collision. Suitable for 3D-scanned concave terrains
    (e.g., a floor with thin walls) where mesh collision is problematic.

    The heightfield is generated by ray-casting from above at each grid point,
    recording the highest intersection with the mesh.
    """

    path: str = MISSING
    """Directory containing terrain STL files and metadata."""

    metadata_yaml: str = MISSING
    """YAML file containing terrain and motion metadata."""

    hfield_resolution: float = 0.02
    """Sampling resolution (meters) for heightfield grid.

    Each grid point fires a downward ray and records the highest intersection.
    Default 0.02 (2 cm) balances accuracy and performance.
    """

    hfield_base_thickness: float = 0.1
    """Fixed base thickness (meters) for the heightfield bottom.

    This is the solid thickness below the lowest point of the terrain.
    A small value (e.g., 0.1m) is sufficient for collision stability.
    """

    hfield_num_workers: int = 0
    """Number of worker processes for parallel ray-casting.

    Set to 0 to use all CPU cores, 1 to disable multiprocessing.
    """

    hfield_use_disk_cache: bool = True
    """If True, persist heightfield arrays to disk for reuse across runs."""

    hfield_cache_dirname: str = ".hfield_visual_cache"
    """Cache directory name created next to each imported STL file."""

    hfield_sink_miss_cells: bool = True
    """If True, sink ray-miss cells deep below terrain instead of interpolating."""

    hfield_floor_z_offset: float = 0.0
    """Z offset applied to align floor surfaces to the same plane.

    Use this to ensure all generated heightfields have their floor at a consistent level.
    """

    hfield_texture_mode: str = "height_color"
    """Texture mode for heightfield visualization: 'height_color' or 'uniform'."""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ):
        return mesh_terrains.stl_heightfield_terrain(self, difficulty, spec, rng)


@dataclass(kw_only=True)
class MotionMatchedTerrainCfg(SubTerrainBaseCfg):
    """Configuration for motion-matched terrain generation.

    ## Terrain Mesh Requirements
    - All terrain meshes must have the a border at the bottom.
    - The terrain origin (0, 0, 0) is at the surface of the terrain center, which means that the point should
        be above the terrain at (0, 0, t) given any t > 0 and below the terrain at (0, 0, t) given any t < 0.
    - The USER should ensure that the non-flat part of the terrain is within the size of the terrain.
    """

    path: str = MISSING
    """Directory containing both terrains and the motions, so that these can be matched together.
    """

    metadata_yaml: str = MISSING
    """YAML file containing the motion matching configuration.
    This file should specify the motion matching parameters, such as the motion files to be used,
    the matching criteria, and any other relevant settings.

    You may use the `scripts/motion_matched_metadata_generator.py` to generate the metadata.yaml file if you arrange your
    dataset in the structure as described in `scripts/motion_matched_metadata_generator.py`.

    ## Typical yaml file structure

    ```yaml
    terrains:
        - terrain_id: "jumpbox1" # can be any string.
          terrain_file: "path/to/terrain.stl" # path to the terrain mesh file, relative to the datasetdir.
        - terrain_id: "jumpbox2"
          terrain_file: "path/to/another_terrain.stl"
    motion_files:
        - terrain_id: "jumpbox1" # should match the terrain_id above.
          motion_file: "path/to/motion1_poses.npz" # path to the motion file, relative to the datasetdir.
          weight: (optional) 1.0
        - terrain_id: "jumpbox2"
          motion_file: "path/to/motion2_retargetted.npz"
          weight: (optional) 1.0
    ```

    """

    collision_hfield: bool = False
    """If True, use a derived MuJoCo hfield for collision while keeping mesh for rendering.

    MuJoCo mesh collision uses convex hulls, which can over-fill non-convex terrain meshes.
    Enabling this keeps visual mesh fidelity while using a heightfield collision surface.
    """

    collision_hfield_resolution: float = 0.01
    """Sampling resolution (meters) used to build the collision hfield from terrain mesh.

    Each grid point fires a downward ray and records the highest intersection with the mesh.
    Default 0.01 (1 cm) gives accurate collision at the cost of a larger hfield array.
    """

    collision_hfield_normal_z_threshold: float = 0.15
    """Only faces with normal_z above this threshold contribute to top-surface sampling.

    Used only by the CoACD auto-align top-surface computation, not by the hfield ray-caster.
    """

    collision_hfield_base_thickness_ratio: float = 1.0
    """Base thickness ratio for the generated collision hfield."""

    collision_hfield_num_workers: int = 0
    """Number of worker processes for parallel hfield ray-casting.

    Set to 0 to use all CPU cores, and 1 to disable multiprocessing (single-process).
    """

    collision_hfield_use_disk_cache: bool = True
    """If True, persist hfield height arrays to disk for reuse across runs.

    The cache is keyed by STL file path, mtime, file size, resolution, and terrain size,
    so it is automatically invalidated when the STL file changes.
    """

    collision_hfield_cache_dirname: str = ".hfield_cache"
    """Cache directory name created next to each imported STL file."""

    collision_hfield_sink_miss_cells: bool = True
    """If True, sink ray-miss cells deep below terrain instead of interpolating/filling.

    Keep this enabled for motion-matched terrains that may intentionally have
    out-of-mesh regions.
    """

    face_box_collision: bool = False
    """If True, replace mesh collision with per-face thin box geoms.

    Root cause: MuJoCo's polyhedral mesh collision model treats the interior of
    any closed mesh as solid.  Objects near the inner surface (e.g. a robot
    standing on the floor inside a scanned room) receive negative-dist contacts
    even when not physically penetrating, causing ``illegal_reset_contact`` to
    fire immediately after spawn.

    Fix: disable collision on the visual mesh geom and add one thin
    ``mjGEOM_BOX`` per triangle face, placed on the outward side of the face.
    Each box only collides from one side, so the robot inside the room is free
    to move without spurious contacts while still colliding correctly with the
    floor and walls.

    This is the recommended option for 3D-scanned room / cup-like meshes.
    Mutually exclusive with ``collision_hfield``.
    """

    face_box_thickness: float = 0.05
    """Thickness (meters) of each per-face box geom for ``face_box_collision``.

    The box extends ``face_box_thickness`` along the face outward normal.
    Larger values give more robust contact detection at the cost of slightly
    thicker invisible walls.
    """

    collision_coacd: bool = False
    """If True, use CoACD approximate convex decomposition for collision geometry.

    This is the recommended option for concave / 3D-scanned meshes (e.g. a
    scanned room or cup-shaped terrain).  MuJoCo's polyhedral mesh collision
    treats the *interior* of any closed mesh as solid, so a robot standing
    inside a scanned room receives spurious contacts.

    CoACD decomposes the mesh into a set of approximate convex hulls.  Each
    hull is added as a separate ``mjGEOM_MESH`` collision geom (invisible,
    group 3) while the original mesh is kept for rendering and depth-camera
    ray-casting (group 2).

    Mutually exclusive with ``collision_hfield`` and ``face_box_collision``.
    """

    collision_coacd_threshold: float = 0.05
    """CoACD concavity threshold (lower = more pieces, better fit).

    Typical range: 0.01 (very fine) – 0.1 (coarse).  Default 0.05 balances
    decomposition quality and the number of resulting convex hulls.
    """

    collision_coacd_max_convex_hull: int = -1
    """Maximum number of convex hulls produced by CoACD (-1 = unlimited)."""

    collision_coacd_preprocess_mode: str = "auto"
    """CoACD pre-processing mode: ``"auto"``, ``"on"``, or ``"off"``."""

    collision_coacd_preprocess_resolution: int = 50
    """Voxel resolution used during CoACD pre-processing."""

    collision_coacd_resolution: int = 2000
    """Sampling resolution used inside the CoACD tree search."""

    collision_coacd_mcts_nodes: int = 20
    """Number of MCTS nodes per iteration in CoACD."""

    collision_coacd_mcts_iterations: int = 150
    """Number of MCTS iterations in CoACD."""

    collision_coacd_mcts_max_depth: int = 3
    """Maximum MCTS tree depth in CoACD."""

    collision_coacd_seed: int = 0
    """Random seed for CoACD (for reproducibility)."""

    collision_coacd_log_level: str = "off"
    """CoACD logger level (e.g. ``\"error\"`` to suppress verbose info output)."""

    collision_coacd_pca: bool = False
    """Whether to enable CoACD PCA pre-processing."""

    collision_coacd_merge: bool = False
    """Whether CoACD merges convex parts after decomposition.

    Keep this False for terrain to reduce collision-vs-visual mismatch ("hover gap")
    introduced by aggressive part merging.
    """

    collision_coacd_decimate: bool = False
    """Whether to decimate each convex hull to max vertex budget."""

    collision_coacd_max_ch_vertex: int = 256
    """Maximum vertices per convex hull when decimation is enabled."""

    collision_coacd_extrude: bool = False
    """Whether to extrude neighboring convex hulls along overlapping faces."""

    collision_coacd_extrude_margin: float = 0.01
    """Extrusion margin used when collision_coacd_extrude=True."""

    collision_coacd_apx_mode: str = "ch"
    """Approximation mode for CoACD: ``\"ch\"`` or ``\"box\"``."""

    collision_coacd_use_disk_cache: bool = True
    """If True, persist CoACD decomposition cache to disk for reuse across runs."""

    collision_coacd_cache_dirname: str = ".coacd_cache"
    """Cache directory name created next to each imported STL file."""

    collision_coacd_prewarm_all: bool = True
    """If True, precompute missing CoACD caches for all terrains in metadata on first use."""

    collision_coacd_prewarm_workers: int = 0
    """Worker-process count for CoACD prewarm.

    Set to 0 to use all CPU cores, and 1 to disable multiprocessing.
    """

    collision_coacd_geom_margin: float = 0.0
    """MuJoCo collision margin assigned to CoACD hull geoms.

    Keep at 0.0 to reduce visible hover gap from collision margin.
    """

    collision_coacd_z_offset: float = 0.0
    """Extra z-offset applied to CoACD collision hull geoms (meters).

    Use a small negative value (e.g. -0.003) if the robot appears to float
    above the rendered terrain surface.
    """

    collision_coacd_auto_align_top_surface: bool = True
    """If True, automatically align CoACD hulls to visual mesh top surface in Z.

    This computes a terrain-specific z correction from the sampled top surface
    of visual mesh and CoACD collision hulls, then applies it to all hull geoms.
    """

    collision_coacd_auto_align_resolution: float = 0.04
    """XY sampling resolution (meters) for CoACD top-surface auto alignment."""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ):
        return mesh_terrains.motion_matched_terrain(self, difficulty, spec, rng)

@dataclass(kw_only=True)
class PerlinMeshFloatingBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a floating box mesh terrain."""

    legacy_function: object = mesh_terrains.floating_box_terrain

    floating_height: tuple[float, float] | float = MISSING
    """The height of the box above the ground. Could be a fixed value or a range (min, max)."""
    box_length: tuple[float, float] | float = MISSING
    """The length of the box along the y-axis. Could be a fixed value or a range (min, max)."""
    box_width: float | None = None
    """The width of the box along the x-axis. If None, it will be equal to the width of the terrain."""
    box_height: tuple[float, float] | float = MISSING
    """The height of the box along the z-axis."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    # values used for perlin noise generation
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = True
    """If True, no perlin noise will be generated exactly below the box."""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng  # legacy implementation uses numpy global RNG internally
        meshes, origin = self.legacy_function(difficulty, self)
        meshes_list = _normalize_legacy_meshes(meshes)
        flat_patches = _sample_legacy_mesh_flat_patches(
            meshes_list=meshes_list,
            origin=np.asarray(origin, dtype=np.float64),
            cfg=self,
        )
        return _legacy_mesh_to_terrain_output(
            meshes=meshes_list,
            origin=origin,
            spec=spec,
            flat_patches=flat_patches,
        )

@dataclass(kw_only=True)
class PerlinMeshRandomMultiBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a sub terrain with multiple random boxes with perlin noise."""

    legacy_function: object = mesh_terrains.random_multi_box_terrain

    box_height_mean: tuple[float, float] | float = MISSING
    box_height_range: float = MISSING
    box_length_mean: tuple[float, float] | float = MISSING
    box_length_range: float = MISSING
    box_width_mean: tuple[float, float] | float = MISSING
    box_width_range: float = MISSING
    platform_width: float = MISSING

    generation_ratio: float = MISSING

    perlin_cfg: PerlinPlaneTerrainCfg | None = None
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = False
    box_perlin_cfg: PerlinPlaneTerrainCfg | None = None
    """Used only when perlin_cfg is not None"""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng  # legacy implementation uses numpy global RNG internally
        meshes, origin = self.legacy_function(difficulty, self)
        meshes_list = _normalize_legacy_meshes(meshes)
        flat_patches = _sample_legacy_mesh_flat_patches(
            meshes_list=meshes_list,
            origin=np.asarray(origin, dtype=np.float64),
            cfg=self,
        )
        return _legacy_mesh_to_terrain_output(
            meshes=meshes_list,
            origin=origin,
            spec=spec,
            flat_patches=flat_patches,
        )
