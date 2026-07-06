import sys
sys.dont_write_bytecode= True
from datetime import datetime


from Config import Config
from State import State
from Tree_Tests import test_clustered, test_accelerations
from QuadTree import build_tree
from Visuals import plot_tree
from Initial_States import initial_state_galaxies, initial_state_cluster_outliers, initial_state_uniform
from Forces import compute_acceleration

cfg = Config(n_particles= 1000,
             collisions= False,
             node_capacity= 1,
             theta= 0.5)

state = State.allocate(cfg= cfg)

initial_state_uniform(state= state, )

build_tree(state= state, 
           cfg= cfg)

test_accelerations(state= state, cfg= cfg)
'''
date=datetime.today().strftime('%Y_%m_%d')

plot_tree(state= state, 
          cfg= cfg, 
          show_particles= True,
          draw_all_bounds= False,
          depth_colour= True,
          save_name=f'Test_{date}.svg')
plot_tree(state= state, 
          cfg= cfg, 
          show_particles= False,
          draw_all_bounds= False,
          depth_colour= True,
          save_name=f'Test_Decomp_{date}.svg')

test_clustered(state= state,
               cfg= cfg,
               n_tests= 10)
'''
print(f'eof ({datetime.today().strftime('%H:%M:%S')})')