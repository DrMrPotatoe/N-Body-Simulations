import sys
sys.dont_write_bytecode= True
import numpy as np

from Config import Config
from State import State

def initial_state_random(state: State, extent= 1, seed= 42) -> None:
    ''' Geneates a random distribuition of points'''
    
    rng = np.random.default_rng(seed= seed)

    n = len(state.particles.x)

    state.particles.x[:] = rng.uniform(low= -extent,
                                       high= extent,
                                       size= n)
    state.particles.y[:] = rng.uniform(low= -extent, 
                                       high= extent,
                                       size= n)
    state.particles.mass[:]= rng.uniform(size= n)
        
def initial_state_uniform(state: State, sigma= 0.25, seed= 42):
    ''' Generates a cluster of points centered around 0.0'''

    rng = np.random.default_rng(seed= seed)

    n = len(state.particles.x)

    state.particles.x[:] = rng.normal(loc= 0.0, 
                                      scale= sigma,
                                      size= n)
    state.particles.y[:] = rng.normal(loc= 0.0, 
                                      scale= sigma, 
                                      size= n)
    state.particles.mass[:]= rng.uniform(size= n)
    
def initial_state_cluster_outliers(state= State, cluster_frac= 0.8, sigma= 0.05, extent= 1, seed= 42):
    ''' Generates a tight cluster around 0, 0 with random points all around'''
    
    rng = np.random.default_rng(seed= seed)

    n = len(state.particles.x)
    n_cluster = int(cluster_frac * n)
    n_random = n - n_cluster

    cluster_x = rng.normal(loc= 0.0, scale= sigma, size= n_cluster)
    cluster_y = rng.normal(loc= 0.0, scale= sigma, size= n_cluster)

    random_x = rng.uniform(low= -extent, high= extent, size= n_random)
    random_y = rng.uniform(low= -extent, high= extent, size= n_random)

    state.particles.x[:] = np.concatenate([cluster_x, random_x])
    state.particles.y[:] = np.concatenate([cluster_y, random_y])
    state.particles.mass[:]= rng.uniform(size= n)

def initial_state_galaxies(state: State, separation= 5, galaxies=3, sigma= 0.3, seed= 42):
    '''Generates 2 clusters of points separation distance away'''
    from utils import split_integer
    rng = np.random.default_rng(seed= seed)

    n = len(state.particles.x)

    n_galaxy_x = []
    n_galaxy_y = []
    n_per_galaxy = split_integer(n, galaxies)
    offset = rng.uniform(high= np.pi)

    for galaxy in range(galaxies):
        n_galaxy = n_per_galaxy[galaxy]

        galaxy_x = separation * np.cos(galaxy / galaxies * 2*np.pi + offset) 
        galaxy_y = separation * np.sin(galaxy / galaxies * 2*np.pi + offset)

        n_galaxy_xi = rng.normal(loc= galaxy_x, scale= sigma, size= n_galaxy)
        n_galaxy_yi = rng.normal(loc= galaxy_y, scale= sigma, size= n_galaxy)

        n_galaxy_x.append(n_galaxy_xi)
        n_galaxy_y.append(n_galaxy_yi)

    state.particles.x[:] = np.concatenate(n_galaxy_x)
    state.particles.y[:] = np.concatenate(n_galaxy_y)
    state.particles.mass[:]= rng.uniform(size= n)

