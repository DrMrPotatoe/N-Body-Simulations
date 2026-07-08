import sys
sys.dont_write_bytecode= True

from State import State
from Config import Config
from QuadTree import build_tree
from Forces import compute_acceleration


def update_acceleration(state: State, cfg: Config):
    ''' re-build the tree and compute the accelerations of each particle'''

    build_tree(state, cfg)
    compute_acceleration(state, cfg)
    
    state.function_calls += 1

def euler_integrator(state: State, cfg: Config):
    ''' Does a single Euler step'''

    update_acceleration(state= state, cfg= Config)

    dt = cfg.dt
    p = state.particles

    p.x += p.vx * dt
    p.y += p.vy * dt

    p.vx += p.ax * dt
    p.vy += p.ay * dt

    state.step += 1

def kdk_integrator(state: State, cfg: Config):
    ''' Does a single kick-drift-kick Step'''

    if state.step == 0:
        update_acceleration(state= state, cfg= cfg)
    dt = cfg.dt
    p = state.particles

    p.vx += 0.5 * p.ax * dt
    p.vy += 0.5 * p.ay * dt

    p.x += p.vx * dt
    p.y += p.vy * dt

    update_acceleration(state= state, cfg= cfg)

    p.vx += 0.5 * p.ax * dt
    p.vy += 0.5 * p.ay * dt

    state.step += 1

integrators = {
    "Euler": euler_integrator,
    "kdk": kdk_integrator,
    }




    



    