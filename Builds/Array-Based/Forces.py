import sys
sys.dont_write_bytecode= True
from math import sqrt
from Config import Config
from State import State

def compute_acceleration(state: State, cfg: Config):
    ''' Computes the acceleration of all (alive) particles'''

    for particle in range(cfg.n_particles):

        if not state.particles.alive[particle]:
            continue

        ax, ay = acceleration_on(state= state,
                                 cfg= cfg,
                                 particle= particle,
                                 node= state.root)
        
        state.particles.ax[particle] = ax
        state.particles.ay[particle] = ay


def acceleration_on(state: State, cfg: Config, particle: int, node: int) -> tuple[float, float]:
    ''' Computes accelerations on particle'''

    ax, ay = 0, 0
    particles = state.particles
    nodes = state.nodes

    if nodes.subtree_particle_count[node] == 0:
        return ax, ay
    
    if nodes.leaf[node]:
        p = nodes.first_particle[node]

        while p != -1:

            if p != particle:

                _ax, _ay = acceleration_between_particles(state= state,
                                                        p1= particle,
                                                        p2= p, 
                                                        cfg= cfg)
                ax += _ax
                ay += _ay

            p = particles.next[p]
    else:
        dx = nodes.x[node] - particles.x[particle]
        dy = nodes.y[node] - particles.y[particle]
        d2 = dx * dx + dy * dy
        s2 = nodes.width[node] ** 2
        t2 = cfg.theta * cfg.theta

        if node != state.root and s2 < t2 * d2:
            return acceleration_between_particle_and_node(state= state,
                                                          p= particle,
                                                          n= node,
                                                          cfg= cfg)
        
        else:
            first_child = nodes.first_child[node]
            for i in range(4):
                _ax, _ay = acceleration_on(state= state,
                                        cfg= cfg,
                                        particle= particle,
                                        node=first_child + i)
                ax += _ax
                ay += _ay
        
    return ax, ay


def compute_acceleration_brute_force(state: State, cfg: Config):
    ''' Computes the acceleration of all (alive) particles '''

    for particle1 in range(cfg.n_particles):
        for particle2 in range(cfg.n_particles):

            if particle1 == particle2:
                continue

            if state.particles.alive[particle1] and state.particles.alive[particle2]:

                ax, ay = acceleration_between_particles(state= state,
                                                        cfg= cfg,
                                                        p1= particle1,
                                                        p2= particle2,)
                
                state.particles.ax[particle1] += ax
                state.particles.ay[particle1] += ay




def acceleration_between_particles(state: State, p1: int, p2: int, cfg: Config) -> tuple[float, float]:
    ''' Force on p1 by p2 particles'''

    particles = state.particles

    dx = particles.x[p2] - particles.x[p1]
    dy = particles.y[p2] - particles.y[p1]

    d2 = dx*dx + dy*dy + (cfg.eps)**2
    df = cfg.G * particles.mass[p2] / (d2 * sqrt(d2))

    ax = df * (dx)
    ay = df * (dy)

    return ax, ay


def acceleration_between_particle_and_node(state: State, p: int, n: int, cfg: Config) -> tuple[float, float]:
    ''' Acceleration between particle and node'''

    particles = state.particles
    nodes = state.nodes

    dx = nodes.mx[n] - particles.x[p]
    dy = nodes.my[n] - particles.y[p]

    d2 = dx*dx + dy*dy + (cfg.eps)**2
    df = cfg.G * particles.mass[p] / (d2 * sqrt(d2))

    ax = df * dx
    ay = df * dy

    return ax, ay