import sys
sys.dont_write_bytecode= True
from datetime import datetime


from Config import Config
from State import State
from Initial_States import initial_state_galaxies, initial_state_cluster_outliers, initial_state_uniform
from Simulate import run_simulation
from QuadTree import build_tree
from Forces import compute_acceleration
from Integrator import integrators
from Tree_Tests import test_clustered, test_accelerations
from Visuals import plot_tree

cfg = Config(n_particles= 10000,
             t_end= 20,
             collisions= False,
             node_capacity= 1,
             theta= 0.5,
             integrator= "kdk",
             video_output_live= True,
             save_frame= False,
             frame_interval= 1,)

integrator = integrators[cfg.integrator]

state = State.allocate(cfg= cfg)

initial_state_uniform(state= state, )

run_simulation(state= state, cfg= cfg, integrator= integrator)
'''
plot_tree(state= state, 
          cfg= cfg, 
          show_particles= True,
          draw_all_bounds= False,
          depth_colour= True,
          save_name=f'Test.svg')
plot_tree(state= state, 
          cfg= cfg, 
          show_particles= False,
          draw_all_bounds= False,
          depth_colour= False,
          save_name=f'Test_Tree_Decomp.svg')

test_accelerations(state= state, cfg= cfg)
test_clustered(state= state,
               cfg= cfg,
               n_tests= 10)
'''
print(f'eof ({datetime.today().strftime('%H:%M:%S')})')