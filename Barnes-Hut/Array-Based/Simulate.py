from collections.abc import Callable
import time
from numpy import count_nonzero

from State import State
from Config import Config
from Escaped_Particles import clean_escaped_particles
from Video_Making import save_frame, init_video, write_video_frame, finish_video
from QuadTree import build_tree
from Integrator import integrators
from Collisions import handle_collisions
from Tree_Statistics import print_tree_status

#type Integrator = Callable[[State, Config], None]

def run_simulation(state: State, cfg: Config,):
    print(f"Simulating {cfg.t_end}s ({cfg.n_steps} steps) of a {cfg.n_particles}-Body System")
    
    start = time.time()
    last_print = start

    video = None

    build_tree(state= state, cfg= cfg)
    integrator = integrators[cfg.integrator]

    if cfg.video_output_live:
        video = init_video(cfg= cfg, state= state)

    frame_id= 0

    if cfg.save_frame:
        save_frame(state= state, 
                   frame_dir= cfg.framedir, 
                   frame_id= frame_id)
        frame_id += 1

    for step in range(cfg.n_steps):

        integrator(state= state, cfg= cfg)

        if cfg.collisions:
            handle_collisions(state= state, cfg= cfg)

        if cfg.escape_factor > 0:
            clean_escaped_particles(state= state, 
                                    cfg= cfg)
            
        if cfg.save_frame:
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

        if cfg.video_output_live:
            write_video_frame(video= video,
                              state= state)


        now = time.time()
        if now - last_print > cfg.progress_update:
            elapsed = now - start
            percent = 100 * step / cfg.n_steps

            print(
                f"\r"
                f"{percent:5.1f}% | "
                f"Step {step:10d} | "
                f"Elapsed {elapsed:12.1f}s | "
                f"Alive {count_nonzero(state.particles.alive):10d} | ",
                end=""
                )

            last_print = now

        if (step+1) % cfg.status_print_interval == 0:
            print_tree_status(state= state, cfg= cfg)

    if cfg.video_output_live:
        finish_video(video= video)
    elapsed = time.time() - start
    print(
        f"\r"
        f"{100:5.1f}% | "
        f"Step {cfg.n_steps:10d} | "
        f"Elapsed {elapsed:12.1f}s",
        )
    print_tree_status(state= state, cfg= cfg)
    

