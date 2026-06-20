import sys
sys.dont_write_bytecode= True

from Config import Config
from State import State

cfg = Config(n_particles= 100,
             collisions= False)

State = State.allocate(cfg= cfg)



print('eof')