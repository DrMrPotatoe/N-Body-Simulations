import sys
sys.dont_write_bytecode= True
import numpy as np

from Config import Config
from State import State
from QuadTree import build_tree, validate_tree
from Initial_States import initial_state_cluster_outliers

def test_clustered(state: State, cfg: Config, n_tests: int = 100, seed: int=  42, cluster_frac: float= 0.8, cluster_spread: float= 0.1, random_spread: float= 5.0) -> None:
    ''' Test for big clusters of points in close proximity'''

    for _ in range(n_tests):

        initial_state_cluster_outliers(state= state, seed= seed)

        build_tree(state= state, cfg= cfg)
        validate_tree(state= state)

    print(f" {n_tests} Cluster Tests passed")