from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True, frozen=True)
class Config:
    """
    Global simulation configuration.

    Immutable after creation.
    """

    # PHYSICS
    G: float = 1e-6
    main_mass: float = 3e4
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
    collisions: bool = True
    remove_escaped_particles: bool = True
    escape_factor: float = 10 # escape radius = factor * log10(n_particles)
    escape_radius: float = 0.0 #Derived

    # OUTPUT
    fps: int = 60
    verbose: int = 0

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