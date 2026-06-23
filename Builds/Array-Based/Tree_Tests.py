import sys
sys.dont_write_bytecode= True
import numpy as np

from Config import Config
from State import State
from QuadTree import build_tree, validate_tree

def test_clustered(state: State, cfg: Config, n_tests: int = 100, seed: int=  42, cluster_frac: float= 0.8, cluster_spread: float= 0.1, random_spread: float= 5.0) -> None:
    ''' Test for big clusters of points in close proximity'''

    rng = np.random.default_rng(seed= seed)

    n_cluster = int(cluster_frac * cfg.n_particles)
    n_random = cfg.n_particles - n_cluster

    cluster_c_x = rng.uniform(-5, 5)
    cluster_c_y = rng.uniform(-5, 5)

    for _ in range(n_tests):

        cluster_x = rng.normal(loc= cluster_c_x, scale= cluster_spread, size= n_cluster)
        cluster_y = rng.normal(loc= cluster_c_y, scale= cluster_spread, size= n_cluster)

        random_x = rng.uniform(low= -random_spread, high= random_spread, size= n_random)
        random_y = rng.uniform(low= -random_spread, high= random_spread, size= n_random)

        state.particles.x[:] = np.concatenate([cluster_x, random_x])
        state.particles.y[:] = np.concatenate([cluster_y, random_y])

        build_tree(state= state, cfg= cfg)
        validate_tree(state= state)

    print(f" {n_tests} Cluster Tests passed")