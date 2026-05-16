from __future__ import annotations
import numpy as np
import math
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from QuadTree import QuadTree
from Point import Point
from Rect import Rect
from Circ import Circ


class QuadTree_Interface:
    '''Test the Quad-Tree implementation'''
    def __init__(self, npoints: int):
        self.capacity = 1
        self.points = []
        self.tree = None
        self.verbose = 1
        self.frame = 0
        self.collide = True

        self.npoints = npoints
        self.theta = 0.5
        self.G = 1e-6
        self.eps = 1e-3
        self.main_mass = 3e4
        self.escape_radius = 4 * np.log10(npoints)
        self.density = 1e6

        self.dt = 0.1
        self.T1 = 10
    
    def create_points(self):
        '''Geneate a random assortment of points'''

        x = np.round(np.random.normal(1, 0.2, (1, self.npoints)).T * 2 -1, 6)
        y = np.round(np.random.normal(1, 0.2, (1, self.npoints)).T * 2 -1, 6)

        #x = np.round(np.random.random_sample((1, npoints)).T * 2 -1, 6)
        #y = np.round(np.random.random_sample((1, npoints)).T * 2 -1, 6)
        
        m = np.random.normal(1, 0.2, (self.npoints, 1))

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()
        w = max(x_max - x_min, y_max - y_min)

        cx = 0.5 * (x_max + x_min)
        cy = 0.5 * (y_max + y_min)

        self.bounds = Rect(cx, cy, w*1.01)

        for i in range(self.npoints):
 
            point = Point(x = x[i].item(), y = y[i].item(), mass = m[i].item(), ID = i)
            point.radius(self.density)
            self.points.append(point)
            #print(f'Point {i}: ({x[i].item()}, {y[i].item()})')

        #print([testmethod.bounds.contains(p) for p in testmethod.points])

    def create_points_orbiting(self):

        from utils import generate_initial_state

        m = np.random.rand(1, self.npoints+1).T
        m[0,0] = self.main_mass
        mpos, mvel = generate_initial_state(self.npoints, mu= self.main_mass * self.G)
        x,y = np.vstack([np.asarray([0,0]).reshape([1,2]),mpos.T]).T
        vx, vy = np.vstack([np.asarray([0,0]).reshape([1,2]),mvel.T]).T

        for i in range(self.npoints+1):

            point = Point(x = x[i].item(), y = y[i].item(), mass = m[i].item(), ID = i)
            point.vx, point.vy = vx[i].item(), vy[i].item()
            point.radius(self.density)
            if i == 0:
                point.r = point.r * 0.001
            self.points.append(point)

    def define_bounds(self):

        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        xmax, xmin = np.max(x), np.min(x)
        ymax, ymin = np.max(y), np.min(y)
        w = max(xmax - xmin, ymax - ymin)

        cx = 0.5 * (xmax + xmin)
        cy = 0.5 * (ymax + ymin)

        self.bounds = Rect(cx, cy, w*1.01)

    def remove_escaped_points(self):
        ''' Removes points that have excaped'''
        escaped2 = self.escape_radius * self.escape_radius

        points_before = len(self.points)
        self.points = [
            p for p in self.points if 
            p.distance2(self.points[0]) < escaped2
        ]
        if self.verbose > 1:
            print(f'Removed points: {points_before - len(self.points)}')

    def build_tree(self):
        '''Builds the tree'''

        self.define_bounds()

        self.tree = QuadTree(bounds= self.bounds,
                             capacity= self.capacity,
                             verbose= self.verbose)
        
        if self.verbose > 1: print(20 * '=')
        for point in self.points:
            self.tree.insert(point)
            if self.verbose > 1:
                print(f'Point {point.id} inserted')
                print(20 * '=')

    def compute_force(self):
        ''' Computes the force of every point'''

        for p in self.points:
            fx, fy = self.tree.force_on(point= p, theta= self.theta, G= self.G, eps= self.eps)
            p.set_force(fx= fx, fy= fy)
            if self.verbose > 2:
                print(p) 

        if self.verbose > 1:
            fx_tot = sum(p.fx for p in self.points)
            fy_tot = sum(p.fy for p in self.points)
            print(f"Net force = ({fx_tot:.6e}, {fy_tot:.6e})")

    def find_collisions(self) -> list:
        ''' Find all collisions in the set of points'''

        collisions = []
        checked = set()

        max_r = np.max([p.r for p in self.points])

        for p in self.points:
            search_area = Circ(p.x, p.y, p.r + max_r)
            
            nearby_points = self.tree.query(search_area)

            for other in nearby_points:
                if other is p:
                    continue
                
                # avoid duplicate pairs 
                pair = tuple(sorted((p.id, other.id)))
                if pair in checked:
                    continue

                checked.add(pair)
                if p.collides(other):
                    collisions.append((p, other))
        
        return collisions
    
    def merge_points(self, p1: Point, p2: Point):
        ''' Merges 2 points into the first'''

        m1, m2 = p1.m, p2.m

        m = m1 + m2

        p1.x = ((p1.x * m1) + (p2.x * m2)) / m
        p1.y = ((p1.y * m1) + (p2.y * m2)) / m

        p1.vx = ((p1.vx * m1) + (p2.vx * m2)) / m
        p1.vy = ((p1.vy * m1) + (p2.vy * m2)) / m

        p1.id = p1.id
        p1.m = m
        p1.radius(self.density)
        if m > self.main_mass -1:
            p1.r = p1.r * 0.001
            
    def collision_handler(self) -> bool:
        ''' Handles collision math and removes collided points
            returns true if any collision happens'''

        collisions = self.find_collisions()

        collision_count = 0
        if len(collisions) == 0:
            return False
        
        removed = set()

        for p1, p2 in collisions:
            if p1 in removed or p2 in removed:
                continue

            self.merge_points(p1, p2)

            removed.add(p2)
            collision_count += 1
            # print(f' Collided {p1.id} and {p2.id}')
        self.points = [p for p in self.points if p not in removed]

        return True
        
    def step(self):
        '''Computes a step'''

        # Update Velocity
        for p in self.points:
            p.half_kick(self.dt)
        
        # Update Pos    
        for p in self.points:
            p.drift(self.dt)

        # Remove escaped particles
        self.remove_escaped_points()

        # Re-compute tree
        self.build_tree()

        if self.collide:
            # do collisions
            collision_check= self.collision_handler()

            # Re-build the tree after collisions
            if collision_check:
                self.build_tree()

        # Re-compute forces
        self.compute_force()

        # Update Velocity
        for p in self.points:
            p.half_kick(self.dt)
        
        self.frame += 1

    def simulate(self):
        ''' Simulates a bunch of steps'''

        steps = int(self.T1 / self.dt)
        for t in range(0, steps):
            
            self.build_tree()

            self.compute_force()

            self.step()

            if self.verbose > 0:
                print(f'T= {t} Done')

    def init_plot(self):
        ''' Initialises the gif plot area'''
        self.fig, self.ax = plt.subplots(figsize=(9,9))

        self.ax.set_aspect('equal')
        self.ax.set_facecolor('k')

        self.scatter = self.ax.scatter([], [], s=[], c= 'white')

        self.ax.set_axis_off()
        self.fig.subplots_adjust(left= 0, bottom= 0, right= 1, top= 1)

        #x0 = self.points[0].x
        #y0 = self.points[0].y
        #w = self.bounds.w
        #self.ax.set_xlim((x0 - w * 3/4), (x0 + w * 3/4))
        #self.ax.set_ylim((y0 - w * 3/4), (y0 + w * 3/4))
        #print('test')

    def gif_animate(self, frame):
        ''' Animates the gif'''
        self.step()
        if self.verbose >-1:
            print(f'Step {self.frame:.0f} Animated')

        x = [p.x for p in self.points]
        y = [p.y for p in self.points]

        m = [float(0)] + [p.m for p in self.points[1:]]
        sz = 2 + 10 * np.log1p(m)

        self.scatter.set_offsets(np.c_[x, y])
        self.scatter.set_sizes(sz)

        x0 = self.points[0].x
        y0 = self.points[0].y
        w = self.escape_radius
        self.ax.set_xlim((x0 - w * 3/8), (x0 + w * 3/8))
        self.ax.set_ylim((y0 - w * 3/8), (y0 + w * 3/8))

        return (self.scatter,)
    
    def gif_simulate(self):
        ''' Simulates the system for the GIF'''
        self.build_tree()
        self.compute_force()

        self.init_plot()

        steps = int(self.T1 / self.dt)

        anim = FuncAnimation(
            self.fig,
            self.gif_animate,
            frames=steps,
            interval=30,
            blit=True
        )

        anim.save(
            "QuadTree.gif",
            writer=PillowWriter(fps=30),
            savefig_kwargs= {
                "facecolor": "black",
                "pad_inches": 0
            }
        )

    def video_simulate(self, filename= 'QuadTree.mp4', fps= 60):
        ''' Simulates the system and saves to video'''

        self.build_tree()
        self.compute_force()
        self.init_plot()

        steps = int(self.T1 / self.dt)

        writer = FFMpegWriter(fps= fps, bitrate=-1)

        with writer.saving(self.fig, outfile= filename, dpi= 300):
            for i in range(steps):

                self.step()

                if i % math.floor(math.sqrt(steps)) == 0:
                    print(f'Frame {i} / {steps} ({len(self.points) } points)')

                x = np.fromiter((p.x for p in self.points), dtype=float)
                y = np.fromiter((p.y for p in self.points), dtype=float)

                m = [0.0] + [p.m for p in self.points[1:]]
                sz = 2 + 10 * np.log1p(m)

                offset = np.zeros((len(x), 2))
                offset[:, 0] = x
                offset[:, 1] = y
                self.scatter.set_offsets(offset)
                self.scatter.set_sizes(sz)

                x0 = self.points[0].x
                y0 = self.points[0].y
                w = self.escape_radius
                self.ax.set_xlim((x0 - w * 3/8), (x0 + w * 3/8))
                self.ax.set_ylim((y0 - w * 3/8), (y0 + w * 3/8))

                writer.grab_frame(facecolor= "black")

    def draw(self, save= True):
        '''Draws the whole tree'''
        fig, ax = plt.subplots(figsize=(9,9))

        self.tree.draw(ax, c='k', lw = 0.5)

        for p in self.points:
            p.draw(ax, size=1)

        ax.set_aspect('equal')
        ax.set_xlim(self.bounds.W, self.bounds.E)
        ax.set_ylim(self.bounds.S, self.bounds.N)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if save == True:
            plt.savefig('quadtree_test.svg', bbox_inches= 'tight')
        else:
            return fig

