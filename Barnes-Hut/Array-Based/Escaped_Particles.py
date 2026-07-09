import numpy as np

from Config import Config
from State import State

def clean_escaped_particles(state: State, cfg: Config):
    ''' Cleans up escaped particles, setting their alive flag to false'''

    particles = state.particles
    root = state.root

    root_x = state.nodes.mx[root]
    root_y = state.nodes.my[root]

    escape2 = cfg.escape_radius * cfg.escape_radius
    
    removed: int = 0

    for i in range(cfg.n_particles):

        if not particles.alive[i]:
            continue
        
        dx = (particles.x[i] - root_x)**2 
        dy = (particles.y[i] - root_y)**2

        if dx + dy > escape2:
            particles.alive[i] = False
            removed += 1

    if cfg.verbose > 2:
        print(f' Removed {removed:3d} Particles in step {state.step}')









'''
def remove_escaped_points(self):
         Removes points that have excaped
        escaped2 = self.escape_radius * self.escape_radius

        points_before = len(self.points)
        self.points = [
            p for p in self.points if 
            p.distance2(self.points[0]) < escaped2
        ]
        if self.verbose > 1:
            print(f'Removed points: {points_before - len(self.points)}')
            
            '''