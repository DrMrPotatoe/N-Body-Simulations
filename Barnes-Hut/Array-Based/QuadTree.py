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

    state.nodes.local_particle_count[last_node : last_node + 4] = 0
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
    state.nodes.local_particle_count[node] = 0

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
    nodes = state.nodes
    particles = state.particles

    nodes.subtree_particle_count[node] += 1
    pm = particles.mass[particle]
    px = particles.x[particle]
    py = particles.y[particle]

    nm = nodes.mass[node]
    nx = nodes.mx[node]
    ny = nodes.my[node]

    nodes.mass[node] += pm
    nodes.mx[node] = ((nx*nm + px*pm) / (nm+pm))
    nodes.my[node] = ((ny*nm + py*pm) / (nm+pm))

    if nodes.leaf[node]:

        particles.next[particle] = nodes.first_particle[node]
        nodes.first_particle[node] = particle

        nodes.local_particle_count[node] += 1

        if nodes.local_particle_count[node] > cfg.node_capacity:

            subdivide(state= state, 
                      node= node,
                      cfg= cfg)
        
        return
    
    quadrant = get_quadrant(state= state, 
                            node= node, 
                            particle= particle)

    child = nodes.first_child[node] + quadrant

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

    state.nodes.local_particle_count[:] = 0
    state.nodes.subtree_particle_count[:] = 0
    state.nodes.leaf[root] = True
    state.nodes.depth[root] = 0

    state.nodes.mx[:] = 0.0
    state.nodes.my[:] = 0.0
    state.nodes.mass[:] = 0.0

    state.node_count = root + 1

    state.particles.ax[:] = 0.0
    state.particles.ay[:] = 0.0

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

        if actual != state.nodes.local_particle_count[node]:

            raise RuntimeError(
                f"Node {node}: "
                f"count={state.nodes.local_particle_count[node]}, "
                f"actual={actual}"
            )
        
    alive_mask = state.particles.alive
    mass_total = sum(state.particles.mass[alive_mask])
    if mass_total != state.nodes.mass[state.root]:
        raise RuntimeError(f"Tree contains {state.nodes.mass[state.root]} mass but it should contain {mass_total}")


def insert_particle_stack(state: State, cfg: Config, particle: int) -> None:
    ''' Inserts a particle into the tree, now iterative'''

    node_stack = state.node_stack
    particle_stack = state.particle_stack

    pmass = state.particles.mass
    nmass = state.nodes.mass

    mx = state.nodes.mx
    my = state.nodes.my
    nx = state.nodes.x
    ny = state.nodes.y
    px = state.particles.x
    py = state.particles.y

    first_child = state.nodes.first_child
    first_particle = state.nodes.first_particle
    next_particle = state.particles.next

    leaf = state.nodes.leaf
    local_count = state.nodes.local_particle_count
    subtree_count = state.nodes.subtree_particle_count

    top = 0

    node_stack[top] = state.root
    particle_stack[top] = particle

    while top >= 0:

        node = node_stack[top]
        particle = particle_stack[top]
        top -=1

        subtree_count[node] += 1

        nm = nmass[node]
        pm = pmass[particle]
        mx[node] = (mx[node]*nm + px[particle]*pm) / (nm + pm)
        my[node] = (my[node]*nm + py[particle]*pm) / (nm + pm)
        nmass[node] += pm

        # if not leaf node
        if not leaf[node]:
            
            #quadrant= get_quadrant(state= state, node= node, particle= particle)
            quadrant = (px[particle]>=nx[node]) + 2 * (py[particle]>= ny[node])

            top += 1
            node_stack[top] = first_child[node] + quadrant
            particle_stack[top] = particle

            continue

        # if leaf has room
        if local_count[node] < cfg.node_capacity:

            next_particle[particle] = first_particle[node]
            first_particle[node] = particle
            local_count[node] += 1

            continue

        # leaf has no room
        # subdivide leaf into particles
        old_particle = subdivide_stack(state= state, node= node)

        # push old particles to the stack

        while old_particle != -1:
            next_old = next_particle[old_particle]
            next_particle[old_particle] = -1

            #quadrant = get_quadrant(state= state, node= node, particle= old_particle)
            quadrant = (px[old_particle]>=nx[node]) + 2 * (py[old_particle]>= ny[node])

            top += 1
            node_stack[top] = first_child[node] + quadrant
            particle_stack[top] = old_particle

            old_particle = next_old

        # push current particle to stack

        #quadrant = get_quadrant(state= state, node= node, particle= particle)
        quadrant = (px[particle]>=nx[node]) + 2 * (py[particle]>= ny[node])

        top += 1
        node_stack[top] = first_child[node] + quadrant
        particle_stack[top] = particle


def subdivide_stack(state: State, node: int) -> int:
    ''' Subdivides a node and returns the first child of the divided node'''

    nodes = state.nodes

    last_node = allocate_children(state= state,
                             parent= node)
    
    nx = nodes.x[node]
    ny = nodes.y[node]
    width = nodes.width[node]

    child_width = width * 0.5
    offset = width * 0.25

    child_offset_key = ((-1, -1),
                        (+1, -1),
                        (-1, +1),
                        (+1, +1)) 
    
    for  i, (dx, dy) in enumerate(child_offset_key):

        child = last_node + i

        nodes.x[child] = nx + dx*offset
        nodes.y[child] = ny + dy*offset
        nodes.width[child] = child_width 

    # Distribute particles

    first_particle = nodes.first_particle[node]

    nodes.first_particle[node] = -1
    nodes.local_particle_count[node] = 0

    return first_particle


def allocate_children_stack(state: State, parent: int) -> None:
    ''' Allocates 4 empty children'''
    nodes = state.nodes

    first_child = state.node_count

    if first_child+4 > len(nodes.x):
        raise BufferError(f'Max node count exceeded ({first_child+4} / {len(nodes.x)})')
    
    state.node_count += 4

    parent_depth = nodes.depth[parent]

    #link parent to children
    nodes.first_child[parent] = first_child
    nodes.leaf[parent] = False

    # Reset children
    children = slice(first_child, first_child + 4)

    nodes.mass[children] = 0.0
    nodes.mx[children] = 0.0
    nodes.my[children] = 0.0

    nodes.first_child[children] = -1
    nodes.first_particle[children] = -1

    nodes.local_particle_count[children] = -1
    nodes.subtree_particle_count[children] = -1

    nodes.leaf[children] = True
    nodes.depth[children] = parent_depth + 1

    return first_child



        





