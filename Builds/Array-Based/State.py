from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from Config import Config

@dataclass(slots= True)
class Particles:
    x: np.ndarray
    y: np.ndarray

    vx: np.ndarray
    vy: np.ndarray

    ax: np.ndarray
    ay: np.ndarray

    mass: np.ndarray
    radius: np.ndarray

    next: np.ndarray
    alive: np.ndarray


@dataclass(slots= True)
class Nodes:
    x: np.ndarray
    y: np.ndarray
    width: np.ndarray
    
    mass: np.ndarray
    mx: np.ndarray
    my: np.ndarray

    first_child: np.ndarray
    first_particle: np.ndarray

    particle_count: np.ndarray
    leaf: np.ndarray
    depth: np.ndarray

@dataclass(slots= True)
class State:
    '''
    Particle and Node information holder
    '''
    particles: Particles
    nodes: Nodes

    root: int
    node_count: int

    time: float
    step: int

    @classmethod
    def allocate(cls, cfg: Config):

        P = Particles(
            x=np.zeros(cfg.n_particles),
            y=np.zeros(cfg.n_particles),

            vx=np.zeros(cfg.n_particles),
            vy=np.zeros(cfg.n_particles),

            ax=np.zeros(cfg.n_particles),
            ay=np.zeros(cfg.n_particles),

            mass=np.zeros(cfg.n_particles),
            radius=np.zeros(cfg.n_particles),

            next=np.full(cfg.n_particles, -1, dtype=np.int32),
            alive=np.ones(cfg.n_particles, dtype=np.bool_)
        )

        N = Nodes(
            x=np.zeros(cfg.max_nodes),
            y=np.zeros(cfg.max_nodes),
            width=np.zeros(cfg.max_nodes),

            mass=np.zeros(cfg.max_nodes),
            mx=np.zeros(cfg.max_nodes),
            my=np.zeros(cfg.max_nodes),

            first_child=np.full(cfg.max_nodes, -1, dtype=np.int32),
            first_particle=np.full(cfg.max_nodes, -1, dtype=np.int32),

            particle_count=np.zeros(cfg.max_nodes, dtype=np.int32),
            leaf=np.ones(cfg.max_nodes, dtype=np.bool_),
            depth=np.zeros(cfg.max_nodes, dtype=np.int32)
        )

        return cls(
            particles=P,
            nodes=N,

            root=0,
            node_count=1,

            time=0.0,
            step=0
        )