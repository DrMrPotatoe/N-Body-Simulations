import sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
from matplotlib import colormaps

from State import State
from Config import Config

def plot_tree(state: State, cfg: Config, 
              show_particles: bool = True,
              show_node_bounds: bool = True,
              draw_all_bounds: bool = True,
              depth_colour: bool = True,
              figsize: tuple[int, int] = (8, 8),
              save_name: str= 'Quadtree_visual_p_b'):
    '''
    Plots the tree, according to the settings (show_particles, show_node_bounds, draw_all_bounds)
    '''

    fig, ax = plt.subplots(figsize= figsize)

    root = state.root

    if show_particles:
        ax.scatter(state.particles.x,
                   state.particles.y,
                   s=5,
                   c= 'k',
                   alpha= 0.5,
                   edgecolors= None,)
        
    if show_node_bounds:

        cmap = colormaps.get_cmap('viridis')

        def draw_node(node: int,):

            is_leaf = state.nodes.leaf[node]

            should_draw = draw_all_bounds or is_leaf
            
            if should_draw:

                half = state.nodes.width[node] * 0.5
                depth = state.nodes.depth[node]

                if depth_colour:
                    colour = cmap(min(depth/ 10, 1.0))
                else:
                    colour = 'k'

                rect = Rectangle(xy= (state.nodes.x[node] - half, state.nodes.y[node] - half),
                                 width= state.nodes.width[node],
                                 height= state.nodes.width[node],
                                 fill= False,
                                 lw= 1 / depth, 
                                 edgecolor= colour,
                                 zorder= depth)
                
                ax.add_patch(rect)

            if not is_leaf:
                first_child = state.nodes.first_child[node]
                draw_node(node= first_child + 0,)
                draw_node(node= first_child + 1,)
                draw_node(node= first_child + 2,)
                draw_node(node= first_child + 3,)
        
        draw_node(node= root,)
        
    half = 0.5 * state.nodes.width[root] 
    padding = 0.1 * state.nodes.width[root]

    ax.set_xlim((state.nodes.x[root] - half - padding),
                (state.nodes.x[root] + half + padding),)
    ax.set_ylim((state.nodes.y[root] - half - padding),
                (state.nodes.y[root] + half + padding),)
    
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.subplots_adjust(left= 0, bottom= 0, right= 1, top= 1)
    plt.tight_layout()
    plt.savefig(f'{cfg.outdir}/{save_name}')
    plt.close(fig= fig)