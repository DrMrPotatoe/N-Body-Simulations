import sys
sys.dont_write_bytecode= True
from datetime import datetime


from Config import Config
from State import State
#from Tree_Tests import test_clustered
from QuadTree import build_tree
from Visuals import plot_tree
from Initial_States import initial_state_galaxies

cfg = Config(n_particles= 1000,
             collisions= False)

state = State.allocate(cfg= cfg)

initial_state_galaxies(state= state,
                       sigma= 1,
                       galaxies= 3)

build_tree(state= state, 
           cfg= cfg)

date=datetime.today().strftime('%Y_%m_%d')
plot_tree(state= state, 
          cfg= cfg, 
          show_particles= True,
          draw_all_bounds= False,
          depth_colour= True,
          save_name=f'Test_{date}.svg')

#test_clustered(state= state,
#               cfg= cfg,
#               n_tests= 10)
sum(state.particles.mass)
print(' eof')