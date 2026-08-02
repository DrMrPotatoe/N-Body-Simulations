import sys
sys.dont_write_bytecode= True
from datetime import datetime


from Config import Config
from State import State
from Initial_States import generate_initial_state
from Simulate import run_simulation
from QuadTree import build_tree
#from Forces import compute_acceleration
from Tree_Tests import test_clustered, test_accelerations
from Visuals import plot_tree
from Tree_Statistics import print_tree_statistics

cfg = Config(n_particles= 1000,
             dt= .1,
             t_end= 10,
             collisions= True,
             remove_escaped_particles= True,
             node_capacity= 1, 
             theta= 0.5,
             integrator= "euler",
             video_output_live= True,
             save_frame= False,
             frame_interval= 1,
             progress_update= 0.1)

state = State.allocate(cfg= cfg)

generate_initial_state(state= state, cfg= cfg)

build_tree(state= state, cfg= cfg)
run_simulation(state= state, cfg= cfg,)

#print_tree_statistics(state= state, cfg= cfg)
#test_accelerations(state=state, cfg= cfg)

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
plot_tree(state= state, 
          cfg= cfg, 
          show_particles= True,
          show_node_bounds= False,
          draw_all_bounds= False,
          depth_colour= False,
          save_name=f'Test_Particles.png')

'''
test_accelerations(state= state, cfg= cfg)

test_clustered(state= state,
               cfg= cfg,
               n_tests= 10)

build_tree(state, cfg)
validate_tree(state)
'''
print(f'eof ({datetime.today().strftime('%H:%M:%S')})')