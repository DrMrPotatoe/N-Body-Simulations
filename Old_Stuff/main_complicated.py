"""
N-body simulator for high N
"""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import numpy.random as rng

def build_parser() ->argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="NBODY", description="Nbody program :)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="Runs the program")
    sp.add_argument("--num", type=int, default=10000, help="Number of bodies")
    sp.add_argument("--t1", type=float, default=3600.0, help="Total integration time")
    sp.add_argument("--dt", type=float, default=1.0, help="Time step")
    sp.add_argument("--dm", type=float, default=100.0, help="Mass difference between main and extra bodies")
    sp.add_argument("--colision", type=bool, default=True, help="Wether to have collisions")
    
    return p

def main(argv = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        body_count = args.num
        time_step = args.dt
        time_end = args.t1
        mass_diff = args.dm
        colisions = args.colision

        body_pos = rng.random([body_count, 2])
        body_mass = [[mass_diff],
                     rng.random([body_count, 1])]

