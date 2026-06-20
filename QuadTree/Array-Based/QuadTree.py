import sys
sys.dont_write_bytecode= True

from Config import Config
from State import State

def allocate_children(state: State, parent: int) -> int:
    '''
    Allocates the empty childen of a node, resetting them, 
    and allocates the first child to the parent. 
    '''

    last_node = state.node_count

    if last_node + 4 > len(state.nodes.x):
        raise RuntimeError('Max n_count exceeded')
    
    state.nodes.first_child[parent] = last_node
    state.nodes.leaf[parent] = False

    # Updaating children:

    state.nodes.mass[last_node : last_node + 4] = 0.0
    state.nodes.mx[last_node : last_node + 4] = 0.0
    state.nodes.my[last_node : last_node + 4] = 0.0

    state.nodes.first_child[last_node : last_node + 4] = -1
    state.nodes.first_particle[last_node : last_node + 4] = -1

    state.nodes.count[last_node : last_node + 4] = 0
    state.nodes.leaf[last_node : last_node + 4] = True

    return last_node + 4


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


def subdivide(state: State, node: int) -> None:
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

        quadrant = get_quadrant(state= state,
                                node= node,
                                particle= p)
        
        child = last_node + quadrant

        state.particles.next[p] = state.nodes.first_particle[child]
        state.nodes.first_particle[child] = p

        state.nodes.particle_count[child] += 1

        p = next_p


def insert_particle(state: State, node: int, particle: int, cfg) -> None:
    pass
        



def reset_tree(state: State) -> None:
    '''
    Resets the node counter, forcing all the particles to go
    back through node 1, forcing the tree to reset whiile keeping
    memory allocations
    '''
    state.node_count = 1
