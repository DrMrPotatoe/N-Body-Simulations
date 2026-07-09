import numpy as np
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from Config import Config
from State import State

def save_frame(state: State, frame_dir: Path, frame_id: int):
    ''' Saves the particle positions, velocity and status for each frame'''

    particles = state.particles

    to_save = np.column_stack((
        particles.x,
        particles.y,
        np.linalg.norm(np.column_stack((particles.vx, particles.vy)), axis=1),
        particles.alive,
        particles.mass
    ))

    np.save(frame_dir / f'frame_{frame_id:06d}.npy', to_save)


@dataclass(slots=True)
class VideoContext:
    fig: Figure
    ax: Axes

    scatter: PathCollection

    writer: FFMpegWriter

    frame: int

    colour_initialised: bool
    vmin: float
    vmax: float

    camera_radius: float


def init_video(cfg: Config, state: State) -> VideoContext:
    """
    Initialises the video writer.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    width = state.nodes.width[state.root]
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.set_axis_off()

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        )

    scatter = ax.scatter(
        [],
        [],
        c=[],
        s=1,
        cmap="plasma",
        )

    limit = max(np.log10(cfg.n_particles) * 0.5, width * 1.2)
    ax.set_xlim(
        -limit,
        limit,
        )

    ax.set_ylim(
        -limit,
        limit,
        )

    writer = FFMpegWriter(
        fps=cfg.fps,
        bitrate=-1,
        )

    writer.setup(
        fig,
        cfg.outdir / cfg.video_filename,
        dpi=300,
    )

    return VideoContext(
        fig=fig,
        ax=ax,
        scatter=scatter,
        writer=writer,
        frame=0,
        colour_initialised=False,
        vmin=0.0,
        vmax=0.0,
        camera_radius= limit
    )


def _draw_frame(
    video: VideoContext,
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    mass: np.ndarray,
):
    """
    Draws one frame and appends it to the video.
    """

    if len(x) == 0:
        return

    video.scatter.set_offsets(
        np.column_stack((x, y))
    )

    video.scatter.set_array(speed)

    if not video.colour_initialised:

        video.vmin = speed.min()
        video.vmax = speed.max()

        video.scatter.set_clim(
            video.vmin,
            video.vmax,
        )

        video.colour_initialised = True

    update_camera(video, x, y, mass)

    video.writer.grab_frame(facecolor="black")

    video.frame += 1


def update_camera(
    video: VideoContext,
    x,
    y,
    mass,
):
    """
    Centres camera on centre of mass.
    """

    m_tot = np.sum(mass)
    x_com = np.sum(x * mass) / m_tot
    y_com = np.sum(y * mass) / m_tot

    r = video.camera_radius

    video.ax.set_xlim(
        x_com-r,
        x_com+r,
    )

    video.ax.set_ylim(
        y_com-r,
        y_com+r,
    )


def write_video_frame(
    video: VideoContext,
    state: State,
):
    """
    Appends the current simulation state to the video.
    """

    alive = state.particles.alive

    x = state.particles.x[alive]
    y = state.particles.y[alive]

    speed = np.sqrt(
        state.particles.vx[alive]**2
        + state.particles.vy[alive]**2
    )

    speed = np.log10(speed + 1e-12)

    mass = state.particles.mass[alive]

    _draw_frame(
        video,
        x,
        y,
        speed,
        mass,
    )


def write_saved_frame(
    video: VideoContext,
    frame: np.ndarray,
):
    """
    Appends a saved .npy frame to the video.
    """

    alive = frame[:, 3].astype(bool)

    _draw_frame(
        video,
        frame[alive, 0],
        frame[alive, 1],
        np.log10(frame[alive, 2] + 1e-12),
    )


def finish_video(
    video: VideoContext,
):
    """
    Finalises the video.
    """

    video.writer.finish()

    plt.close(video.fig)


def make_video_from_frames(
    cfg: Config,
    state: State,
    frame_dir: Path,
):
    """
    Creates a video from saved .npy frames.
    """

    video = init_video(cfg, state)

    frame_files = sorted(
        frame_dir.glob("frame_*.npy")
    )

    total = len(frame_files)

    for i, file in enumerate(frame_files):

        frame = np.load(file)

        write_saved_frame(
            video,
            frame,
        )

        if i % max(1, total // 20) == 0:
            print(
                f"Frame {i+1}/{total}"
            )

    finish_video(video)