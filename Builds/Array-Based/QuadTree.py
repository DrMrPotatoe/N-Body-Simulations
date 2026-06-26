import sys
sys.dont_write_bytecode= True
import numpy as np

from Config import Config
from State import State

def allocate_children(state: State, parent: int) -> int:
    '''
    Allocates the empty childen of a node, resetting them, 
    and allocates the first child to the parent. 
    '''

    last_node = state.node_count
    parent_depth = state.nodes.depth[parent]
    

    if last_node + 4 > len(state.nodes.x):
        raise RuntimeError(f'Max n_count exceeded ({last_node + 4}, p= {state.nodes.first_particle[parent]})')
    
    state.node_count += 4

    state.nodes.first_child[parent] = last_node
    state.nodes.leaf[parent] = False

    # Updating children:

    state.nodes.mass[last_node : last_node + 4] = 0.0
    state.nodes.mx[last_node : last_node + 4] = 0.0
    state.nodes.my[last_node : last_node + 4] = 0.0

    state.nodes.first_child[last_node : last_node + 4] = -1
    state.nodes.first_particle[last_node : last_node + 4] = -1

    state.nodes.particle_count[last_node : last_node + 4] = 0
    state.nodes.leaf[last_node : last_node + 4] = True
    state.nodes.depth[last_node : last_node + 4] = parent_depth + 1

    return last_node 


def get_quadrant(state: State, node: int, particle: int) -> int:
    '''
    returns the Quadrant of a particle within a node:
    0- SW;
    1- SE;
    2- NW;
    3- NE;
    '''
    px = state.particles.x[particle]
    py = state.particles.y[particle]

    nx = state.nodes.x[node]
    ny = state.nodes.y[node]

    east = px >= nx
    north= py >= ny

    return east + 2 * north


def subdivide(state: State, node: int, cfg: Config) -> None:
    '''
    Subdivides a node into its children, and allocates the particles accordignly.
    '''

    # Allocating childs

    last_node = allocate_children(state= state,
                             parent= node)
    
    nx = state.nodes.x[node]
    ny = state.nodes.y[node]
    width = state.nodes.width[node]

    child_width = width * 0.5
    offset = width * 0.25

    child_offset_key = ((-1, -1), 
                        (+1, -1),
                        (-1, +1),
                        (+1, +1))
    
    for  i, (dx, dy) in enumerate(child_offset_key):

        child = last_node + i

        state.nodes.x[child] = nx + dx*offset
        state.nodes.y[child] = ny + dy*offset

        state.nodes.width[child] = child_width 

    # Distribute particles

    p = state.nodes.first_particle[node]

    state.nodes.first_particle[node] = -1
    state.nodes.particle_count[node] = 0

    while p != -1:

        next_p = state.particles.next[p]
        state.particles.next[p] = -1

        quadrant = get_quadrant(state= state,
                                node= node,
                                particle= p)
        
        child = last_node + quadrant

        insert_particle(state= state,
                        node= child,
                        particle= p,
                        cfg= cfg)

        p = next_p


def insert_particle(state: State, node: int, particle: int, cfg: Config) -> None:
    '''
    Inserts particles into the appropriate quadrant, and orders a subdivide if needed
    '''

    pm = state.particles.mass[particle]
    px = state.particles.x[particle]
    py = state.particles.y[particle]

    nm = state.nodes.mass[particle]
    nx = state.nodes.mx[particle]
    ny = state.nodes.my[particle]

    state.nodes.mass[node] += pm
    state.nodes.mx[node] = ((nx*nm + px*pm) / (nm+pm))
    state.nodes.my[node] = ((ny*nm + py*pm) / (nm+pm))

    if state.nodes.leaf[node]:

        state.particles.next[particle] = state.nodes.first_particle[node]
        state.nodes.first_particle[node] = particle

        state.nodes.particle_count[node] += 1

        if state.nodes.particle_count[node] > cfg.node_capacity:

            subdivide(state= state, 
                      node= node,
                      cfg= cfg)
        
        return
    
    quadrant = get_quadrant(state= state, 
                            node= node, 
                            particle= particle)

    child = state.nodes.first_child[node] + quadrant

    insert_particle(state= state, 
                    node= child, 
                    particle= particle, 
                    cfg= cfg)


def build_tree(state: State, cfg: Config) -> None:
    '''
    Builds the tree from the state and config data classes
    '''
    reset_tree(state= state)

    for p in range(cfg.n_particles):

        if not state.particles.alive[p]:
            continue

        insert_particle(state= state,
                        node= state.root,
                        particle= p,
                        cfg= cfg)


def reset_tree(state: State) -> None:
    '''
    Resets the node counter, forcing all the particles to go
    back through node 1, forcing the tree to reset whiile keeping
    memory allocations
    '''

    root = state.root
  
    state.nodes.first_child[root] = -1
    state.nodes.first_particle[root] = -1

    state.nodes.particle_count[root] = 0
    state.nodes.leaf[root] = True
    state.nodes.depth[root] = 0

    state.nodes.mx[:] = 0.0
    state.nodes.my[:] = 0.0
    state.nodes.mass[:] = 0.0


    state.node_count = root + 1

    initialize_root(state= state)


def initialize_root(state: State) -> None:
    ''' 
    Initialises the root node, with a width determined by particles.
    '''
    alive = state.particles.alive

    x = state.particles.x[alive]
    y = state.particles.y[alive]

    xmin = np.min(x)
    xmax = np.max(x)

    ymin = np.min(y)
    ymax = np.max(y)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    width = max(xmax - xmin, ymax - ymin, 1e-3) * 1.1

    root = state.root

    state.nodes.x[root] = cx
    state.nodes.y[root] = cy

    state.nodes.width[root] = width


def validate_tree(state: State) -> None:
    '''
    Validates the tree. Includes:

    appearance;
    node count;
    mass;
    
    '''
    # Each particle appears once in the tree

    seen = np.zeros(len(state.particles.x), dtype= bool)

    total = 0

    for node in range(state.root, state.node_count):
        
        if not state.nodes.leaf[node]:
            continue

        p = state.nodes.first_particle[node]

        while p != -1:

            if seen[p]:
                raise RuntimeError(f" Particle {p} has multiple appearances")
            
            seen[p] = True
            total += 1

            p = state.particles.next[p]

    alive = np.count_nonzero(state.particles.alive)

    if total != alive:
        raise RuntimeError(f"Tree contains {total} particles but {alive} are alive")
    
    # Node Counts

    for node in range(state.root, state.node_count):

        if not state.nodes.leaf[node]:
            continue

        actual = 0

        p = state.nodes.first_particle[node]

        while p != -1:

            actual += 1
            p = state.particles.next[p]

        if actual != state.nodes.particle_count[node]:

            raise RuntimeError(
                f"Node {node}: "
                f"count={state.nodes.particle_count[node]}, "
                f"actual={actual}"
            )
        
    alive_mask = state.particles.alive
    mass_total = sum(state.particles.mass[alive_mask])
    if mass_total != state.nodes.mass[state.root]:
        raise RuntimeError(f"Tree contains {state.nodes.mass[state.root]} mass but it should contain {mass_total}")



