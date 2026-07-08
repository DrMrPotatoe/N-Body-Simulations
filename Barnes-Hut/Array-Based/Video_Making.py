import sys
sys.dont_write_bytecode= True
import numpy as np
from pathlib import Path

from Config import Config
from State import State

def save_frame(state: State, frame_dir: Path, frame_id: int):
    ''' Saves the particle positions, velocity and status for each frame'''

    particles = state.particles

    to_save = np.column_stack((
        particles.x,
        particles.y,
        np.linalg.norm(np.column_stack((particles.vx, particles.vy)), axis=1),
        particles.alive
    ))

    np.save(frame_dir / f'frame_{frame_id:06d}.npy', to_save)
