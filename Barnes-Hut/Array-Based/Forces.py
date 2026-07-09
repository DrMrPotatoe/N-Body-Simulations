from math import sqrt
import numpy as np
from Config import Config
from State import State

def compute_acceleration(state: State, cfg: Config):
    ''' Computes the acceleration of all (alive) particles'''

    particles = state.particles
    nodes = state.nodes
    

    data = {
        "px": particles.x,
        "py": particles.y,
        "pmass": particles.mass,
        "next": particles.next,

        "mx": nodes.mx,
        "my": nodes.my,
        "mass": nodes.mass,
        "width": nodes.width,
        "leaf": nodes.leaf,
        "first_particle": nodes.first_particle,
        "first_child": nodes.first_child,
        "subtree_count": nodes.subtree_particle_count,

        "eps2": cfg.eps**2,
        "theta2": cfg.theta**2,
        "G": cfg.G,
        "root": state.root,

        "stack": np.empty(cfg.max_nodes, dtype= np.int32)
    }

    local_interactions = 0
    local_visits = 0

    for p in range(cfg.n_particles):

        if not particles.alive[p]:
            continue
        
        ax, ay, interactions, visits = acceleration_on(data= data,
                                 particle= p,)
        
        particles.ax[p] = ax
        particles.ay[p] = ay

        local_interactions += interactions 
        local_visits += visits

    state.particle_interactions += local_interactions
    state.node_visits += local_visits



def acceleration_on(data: dict, particle: int,) -> tuple[float, float, int]:
    ''' Computes accelerations on particle, now with a stack!'''

    # Cache arrays
    px = data["px"]
    py = data["py"]
    pmass = data["pmass"]
    next_particle = data["next"]

    mx = data["mx"]
    my = data["my"]
    nmass = data["mass"]
    width = data["width"]
    leaf = data["leaf"]
    first_particle = data["first_particle"]
    first_child = data["first_child"]
    subtree_count = data["subtree_count"]

    # Constants
    eps2 = data["eps2"]
    theta2 = data["theta2"]
    G = data["G"]
    root = data["root"]
    stack = data["stack"]

    x = px[particle]
    y = py[particle]

    ax = 0.0
    ay = 0.0

    interactions = 0
    node_visits = 0

    top = 0
    stack[top] = root
    top += 1

    while top:

        top -= 1
        node = stack[top]
        node_visits += 1

        if subtree_count[node] == 0:
            continue

        # If Leaf:
        if leaf[node]:

            p = first_particle[node]

            while p != -1:

                if p != particle:

                    dx = px[p] - x
                    dy = py[p] - y
                    d2 = dx*dx + dy*dy + eps2

                    invr3 = 1.0 / (d2 * sqrt(d2))
                    df = G * pmass[p] * invr3

                    ax += df * dx
                    ay += df * dy

                    interactions += 1
                
                p = next_particle[p]

        # If Node
        else:
            dx = mx[node] - x
            dy = my[node] - y
            d2 = dx*dx + dy*dy + eps2

            if node != root and width[node]**2 < theta2*d2:
                
                invr3 = 1.0 / (d2 * sqrt(d2))
                df = G * nmass[node] * invr3

                ax += df * dx
                ay += df * dy

                interactions += 1

            else:
                child = first_child[node]
                # add to stack
                stack[top] = child
                stack[top+1] = child+1
                stack[top+2] = child+2
                stack[top+3] = child+3
                top += 4

    return ax, ay, interactions, node_visits

                




    
def acceleration_between_particles(state: State, p1: int, p2: int, cfg: Config) -> tuple[float, float]:
    ''' Force on p1 by p2 particles'''

    particles = state.particles

    dx = particles.x[p2] - particles.x[p1]
    dy = particles.y[p2] - particles.y[p1]

    d2 = dx*dx + dy*dy + (cfg.eps)**2
    df = cfg.G * particles.mass[p2] / (d2 * sqrt(d2))

    ax = df * (dx)
    ay = df * (dy)

    state.particle_interactions += 1

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

    state.particle_interactions += 1

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


"""
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


def acceleration_between_particles(state: State, p1: int, p2: int, cfg: Config) -> tuple[float, float]:
    ''' Force on p1 by p2 particles'''

    particles = state.particles

    dx = particles.x[p2] - particles.x[p1]
    dy = particles.y[p2] - particles.y[p1]

    d2 = dx*dx + dy*dy + (cfg.eps)**2
    df = cfg.G * particles.mass[p2] / (d2 * sqrt(d2))

    ax = df * (dx)
    ay = df * (dy)

    state.particle_interactions += 1

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

    state.particle_interactions += 1

    return ax, ay
"""