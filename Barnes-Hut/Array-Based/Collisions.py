#import numpy as np
from numpy import pi, clip, hypot, exp
from math import sqrt

from Config import Config
from State import State

def assign_radius(state: State, cfg: Config) -> None:
    ''' Work out and assing the radius of every (alive) particle'''

    mass = state.particles.mass
    alive = state.particles.alive
    density = cfg.density
    radius = state.particles.radius

    for p in range(cfg.n_particles):
        if not alive[p]:
            continue

        if mass[p] > sqrt(cfg.main_mass):
            radius[p] = pow((3 * mass[p])/(4*pi*pow(density,2)), 1/3)
        else: 
            radius[p] = pow((3 * mass[p])/(4*pi*density), 1/3)
            
    state.particles.max_radius = radius.max()


def find_collisions(state: State, cfg: Config) -> None:
    ''' Finds the closest particle to each (alive) particle and returns it iinside the particle_stack'''

    nodes = state.nodes
    particles = state.particles
    stack = state.node_stack
    state.particle_stack.fill(-1)
    closest_stack = state.particle_stack

    nx = nodes.x
    ny = nodes.y
    width = nodes.width
    leaf = nodes.leaf
    first_particle = nodes.first_particle
    first_child = nodes.first_child

    px = particles.x
    py = particles.y
    alive = particles.alive
    radius = particles.radius
    max_radius = particles.max_radius
    next_particle = particles.next
    
    for p in range(cfg.n_particles):

        if not alive[p]:
            continue
        
        x = px[p]
        y = py[p]
        r = radius[p]
        r_max = r + max_radius

        closest: int = -1
        closest_dist2: int = cfg.n_particles

        top = 0
        stack[top] = state.root
        top += 1

        while top:

            top -= 1
            node = stack[top]

            width_2 = width[node] * width[node]
            left_edge = nx[node] - width_2
            right_edge = nx[node] + width_2
            bottom_edge = ny[node] - width_2
            top_edge = ny[node] + width_2

            dx = x - clip(x, left_edge, right_edge)
            dy = y - clip(y, bottom_edge, top_edge)

            if dx*dx + dy*dy > r_max * r_max:
                continue

            if leaf[node]:
                
                node_p = first_particle[node]

                while node_p != -1:
                    if node_p > p: #avoids looking back

                        dx = px[node_p] - x
                        dy = py[node_p] - y

                        dist2 = dx*dx + dy*dy
                        dr2 = (r + radius[node_p]) * (r + radius[node_p])

                        if dist2 < dr2 and dist2 < closest_dist2:
                            closest = node_p
                            closest_dist2 = dist2
                    
                    node_p = next_particle[node_p]
            else:

                child = first_child[node]
                stack[top] = child
                stack[top+1] = child+1
                stack[top+2] = child+2
                stack[top+3] = child+3
                top += 4


        closest_stack[p] = closest


def merge_collision(state: State, i: int, j: int, cfg: Config) -> None:
    ''' Merges 2 particles to the first'''

    particles = state.particles

    x = particles.x
    y = particles.y
    
    vx = particles.vx
    vy = particles.vy

    m1 = particles.mass[i]
    m2 = particles.mass[j]

    m_new = m1 + m2

    x[i] = (x[i]*m1 + x[j]*m2) / m_new 
    y[i] = (y[i]*m1 + y[j]*m2) / m_new

    vx[i] = (vx[i]*m1 + vx[j]*m2) / m_new 
    vy[i] = (vy[i]*m1 + vy[j]*m2) / m_new 

    particles.alive[j] = False


def inelastic_collision(state: State, i: int, j: int, cfg: Config) -> None:
    ''' collides the particles inelastically depending on the restituition coefficient'''

    particles = state.particles

    x = particles.x
    y = particles.y

    vx = particles.vx
    vy = particles.vy

    mass = particles.mass
    e = cfg.restitution_coefficient

    dx = x[j] - x[i]
    dy = y[j] - y[i]

    dr = sqrt(dx*dx + dy*dy)

    nx = dx / dr
    ny = dx / dr

    dvx = vx[j] - vx[i]
    dvy = vy[j] - vy[i]

    rel = dvx*nx + dvy*ny

    inv_mass = 1/mass[i] + 1/mass[j]

    J = -(1 + e) * rel / inv_mass

    vx[i] -= J * nx / mass[i]
    vy[i] -= J * ny / mass[i]

    vx[j] += J * nx / mass[j]
    vy[j] += J * ny / mass[j]


def mass_transfer_collision(state: State, i: int, j: int, cfg: Config) -> None:
    ''' Collides 2 particles and transfers mass to the heavier one according to the collision's stats'''

    particles = state.particles

    x = particles.x
    y = particles.y
    
    vx = particles.vx
    vy = particles.vy

    mass = particles.mass

    if mass[i] > mass[j]:
        p1 = i
        p2 = j
    else:
        p1 = j
        p2 = i

    m1 = particles.mass[p1]
    m2 = particles.mass[p2]

    v_rel = hypot(vx[p1]-vx[p2], vy[p1]-vy[p2])

    mu = (m1 * m2) / (m1 + m2)

    eta = exp(-v_rel / cfg.collision_velocity)

    dm = eta * mu

    dm = min(dm, m2)

    if m2 - dm < cfg.minimum_mass_fraction * cfg.starting_mass:
        dm = m2
        particles.alive[p2] = False

    mass[p1] += dm

    vx[p1] = ((vx[p1]*m1) +(vx[p2]*dm)) / mass[p1]
    vy[p1] = ((vy[p1]*m1) +(vy[p2]*dm)) / mass[p1]

    if particles.alive[p2]:
        mass[p2] - dm


def handle_collisions(state: State, cfg: Config):
    ''' assigns radius and handles all the collisions for this timestep'''
    collision = collision_modes[cfg.collision_type]

    assign_radius(state= state, cfg= cfg)

    find_collisions(state= state, cfg= cfg)

    particles = state.particles

    alive = particles.alive
    collisions = state.particle_stack

    for i in range(cfg.n_particles):

        if not alive[i]:
            continue

        j = collisions[i]

        if j == -1 or j < i or not alive[j]: 
            continue

        collision(state= state,
                  i= i, 
                  j=j, 
                  cfg= cfg)


collision_modes = {
    "merge": merge_collision,
    "inelastic": inelastic_collision,
    "mass_transfer": mass_transfer_collision,
    }





            




