from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path

@dataclass(slots=True, frozen=True)
class Config:
    """
    Global simulation configuration.
    """

    # PHYSICS
    G: float = 1e-6
    main_mass: float = 3e4
    starting_mass: float = 1
    main_bodies: int = 1
    density: float = 1e6
    eps: float = 1e-3

    # BARNES HUT AND TREE
    n_particles: int = 1000
    theta: float = 0.5
    max_nodes: int | None = None # If None, chosen automatically
    node_capacity: int = 1

    # SIMULATION
    dt: float = 0.1
    t_end: float = 10.0
    n_steps: int = 0 #Derived
    remove_escaped_particles: bool = False
    escape_factor: float = 3 # escape radius = factor * log10(n_particles)
    escape_radius: float = 0.0 #Derived; disable by scale_factor= 0
    integrator: str = "euler"

    # COLLISIONS
    collisions: bool = False
    collision_type: str = "merge"
    restitution_coefficient: float = 0.5
    collision_velocity: float = 0.5
    minimum_mass_fraction: float = 0.05

    # OUTPUT
    video_output_end: bool = False
    video_output_live: bool = True
    save_frame: bool = True
    fps: int = 60
    frame_interval: int = 3 # How many steps to do between every frame of the output
    verbose: int = 0
    progress_update: float = 1 # How many seconds to wait between updates to the progress bar
    status_print_interval = 25 # How many steps between tree status summary
    outdir: Path = Path('./outputs')
    framedir: Path = Path('./frames')
    video_filename: Path = Path('Barnes_Hut_out.mp4')

    def __post_init__(self):

        n_steps = int(math.ceil(self.t_end / self.dt))

        if self.max_nodes is None:
            max_nodes = 8 * self.n_particles
        else:
            max_nodes = self.max_nodes

        escape_radius = (
            self.escape_factor *
            math.log10(max(self.n_particles, 10))
        )

        # Dataclass updates
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "max_nodes", max_nodes)
        object.__setattr__(self, "escape_radius", escape_radius)

        # Other updates:
        Path(self.outdir).mkdir(parents=True, exist_ok= True)
        Path(self.framedir).mkdir(parents=True, exist_ok= True)