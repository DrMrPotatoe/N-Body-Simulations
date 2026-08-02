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
    density: float = 1e6
    eps: float = 1e-3
    n_particles: int = 1000
    dt: float = 0.1
    t_end: float = 10.0
    n_steps: int = 0 #Derived
    integrator: str = "euler"

    #INITIAL CONDITIONS
    main_mass: float = 3e4
    minor_mass: float = 1
    mass_sigma: float = 0.5
    main_bodies: int = 5
    n_clusters: int = 2 # Number of clusters used for the initial conditions. 0= random 1=normal
    cluster_sigma: float = 1 # sigma of every cluster
    speed_variation: float = 0.1 #Variation in orbiting speed. done to avoid only circular orbits
    cluster_separation: float = 3
    outlier_ratio: float = 0.0 # fraction of points randomly distribuited (0-1) 
    initiate_orbiting: bool = True # if particles are generated with orbiting velocity (True, False)
    generation_extent: float = 10 # Area in which the particles will be generated
    generation_seed: float = 42 # RNG seed

    # BARNES HUT AND TREE
    theta: float = 0.5
    max_nodes: int | None = None # If None, chosen automatically
    node_capacity: int = 1

    # COLLISIONS AND CLEANUP
    collisions: bool = False
    collision_type: str = "merge"
    restitution_coefficient: float = 0.5
    collision_velocity: float = 0.5
    minimum_mass_fraction: float = 0.05
    remove_escaped_particles: bool = False
    escape_factor: float = 3 # escape radius = factor * generation_extent
    escape_radius: float = 0.0 #Derived; disable by scale_factor= 0

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

        escape_radius = (self.escape_factor * self.generation_extent)

        # Dataclass updates
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "max_nodes", max_nodes)
        object.__setattr__(self, "escape_radius", escape_radius)

        # Other updates:
        Path(self.outdir).mkdir(parents=True, exist_ok= True)
        Path(self.framedir).mkdir(parents=True, exist_ok= True)