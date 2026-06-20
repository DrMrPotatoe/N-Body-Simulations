import sys
sys.dont_write_bytecode= True

from Config import Config
from State import State
from Tree_Tests import test_clustered

cfg = Config(n_particles= 1000,
             collisions= False)

state = State.allocate(cfg= cfg)

test_clustered(state= state,
               cfg= cfg,
               n_tests= 10)

print(' eof')