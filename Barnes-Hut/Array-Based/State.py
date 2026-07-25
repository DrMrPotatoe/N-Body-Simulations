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
    max_radius: float

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

    local_particle_count: np.ndarray
    subtree_particle_count: np.ndarray
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

    step: int

    function_calls: int
    p2p_interactions: int
    p2n_interactions: int
    node_visits: int

    move_time: float
    force_time: float
    tree_time: float

    particle_stack: np.ndarray
    node_stack: np.ndarray

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
            max_radius=0.0,

            next=np.full(cfg.n_particles, -1, dtype=np.int32),
            alive=np.ones(cfg.n_particles, dtype=np.bool_),

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

            local_particle_count=np.zeros(cfg.max_nodes, dtype=np.int32),
            subtree_particle_count= np.zeros(cfg.max_nodes, dtype= np.int32),
            leaf=np.ones(cfg.max_nodes, dtype=np.bool_),
            depth=np.zeros(cfg.max_nodes, dtype=np.int32)
        )

        return cls(
            particles=P,
            nodes=N,

            root=0,
            node_count=1,

            step=0,

            function_calls=0,
            p2p_interactions=0,
            p2n_interactions=0,
            node_visits=0,

            move_time= 0,
            tree_time= 0,
            force_time= 0,

            particle_stack = np.empty(cfg.max_nodes, dtype= np.int32),
            node_stack = np.empty(cfg.max_nodes, dtype= np.int32),

        )