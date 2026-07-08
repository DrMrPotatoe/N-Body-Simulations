import sys
sys.dont_write_bytecode= True
from collections.abc import Callable
import time

from State import State
from Config import Config
from Escaped_Particles import clean_escaped_particles
from Video_Making import save_frame

type Integrator = Callable[[State, Config], None]

def run_simulation(state: State, cfg: Config, integrator: Integrator):
    print(f"Simulating {cfg.t_end}s ({cfg.n_steps} steps) of a {cfg.n_particles}-Body System")
    
    start = time.time()
    last_print = start

    frame_id= 0

    if cfg.video_output:
        save_frame(state= state, 
                    frame_dir= cfg.framedir, 
                    frame_id= frame_id)
        frame_id += 1
    for step in range(cfg.n_steps):

        integrator(state= state, cfg= cfg)

        if cfg.collisions:
            pass

        if cfg.escape_factor > 0:
            clean_escaped_particles(state= state, 
                                    cfg= cfg)

        if cfg.video_output:
            if step == cfg.n_steps - 1:
                save_frame(state= state, 
                           frame_dir= cfg.framedir, 
                           frame_id= frame_id)
                frame_id += 1
            
            elif (step+1) % cfg.frame_interval == 0:
                save_frame(state= state,
                           frame_dir= cfg.framedir, 
                           frame_id= frame_id)
                
                frame_id += 1



        now = time.time()
        if now - last_print > cfg.progress_update:
            elapsed = now - start
            percent = 100 * step / cfg.n_steps

            print(
                f"\r"
                f"{percent:5.1f}% | "
                f"Step {step + 1:10d} | "
                f"Elapsed {elapsed:12.1f}s",
                end=""
                )

            last_print = now
    elapsed = time.time() - start
    print(
        f"\r"
        f"{100:5.1f}% | "
        f"Step {cfg.n_steps:10d} | "
        f"Elapsed {elapsed:12.1f}s",
        )

