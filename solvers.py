import numpy as np
from utils import generate_initial_state

class N_Body_Solver:
    def __init__(self, N_Bodies: np.ndarray, end_time: float, dT: float|None, save_dT: int = 10, plot: bool = False):
        self.iNum_Bodies = N_Bodies
        self.bPlot = plot
        self.fEnd_Time = end_time
        self.fTime_Step = dT
        self.idTperFrame = save_dT
        self.bCollisions = False
        self.fDensity = 0.1

        self.fGravity_Constant = 1.0E-9
        self.fMain_Mass = 10000  
        self.theta = 0.5 # Quadtree threshold

    def prepare(self):
        self.vMass_Bodies = np.random.rand(1, self.iNum_Bodies+1).T
        self.vMass_Bodies[0,0] = self.fMain_Mass
        mpos, mvel = generate_initial_state(self.iNum_Bodies, mu= self.fMain_Mass * self.fGravity_Constant)
        self.mPos_Bodies = np.vstack([np.asarray([0,0]).reshape([1,2]),mpos.T])
        self.mVel_Bodies = np.vstack([np.asarray([0,0]).reshape([1,2]),mvel.T])

        self.vRadius_bodies = ((self.vMass_Bodies*self.fDensity)/(np.pi))**(1/2)
        self.vRadius_bodies[0] = 0.01
        print(f"Initialised {self.iNum_Bodies} objects \n")

    @staticmethod
    def naive_accel_step(npos, nmass, nrad, G):
        nbodies = np.size(nmass)

        accel = np.zeros_like(npos)
        for i in range(0, nbodies):
            ai = 0
            for j in range(0, nbodies):
                if i == j:
                    continue    
                Rij = npos[i, :] - npos[j, :] # Distance between bodies
                Rij_norm = np.linalg.norm(Rij) # Norm of distance          
                Rij_norm = np.maximum(Rij_norm, nrad[i]) # preventing singularities
                inv_Rij3 = 1.0 / (Rij_norm**3) #1/rij^3
                ai += - G * nmass[j] * Rij * inv_Rij3 #gettin acceleration
            accel[i, :] = ai
        return accel
    
    @staticmethod
    def semivector_accel_step(npos, nmass, nrad, G):
        nbodies = np.size(nmass)
        accel = np.zeros_like(npos)
        for i in range(0, nbodies):
            Rij = npos[i, :] - npos
            Rij_norm = np.linalg.norm(Rij, axis=1)
            Rij_norm = np.where(Rij_norm[:, None] < nrad[i] + nrad, nrad,  Rij_norm[:, None]) # preventing singularities
            Rij_norm[i] = np.nan
            inv_Rij3 = 1.0 / (Rij_norm**3) #1/rij^3
            ai = - G * np.nansum((nmass * inv_Rij3) * Rij, axis= 0) #gettin acceleration
            accel[i, :] = ai
        return accel

    def quadtree_accel_step(npos, nmass, nad, G, theta):
        accel = npos
        return accel
    
    def quadtree_insert():
        accel = 1
        return accel

    def quadtree_traverse():
        accel = 1
        return accel
    
    def build_quadtree():
        accel = 1
        return accel




    def test(self):
        # Testing naive accel step
        # return self.naive_accel_step(self.mPos_Bodies, self.vMass_Bodies, self.vRadius_bodies, self.fGravity_Constant)
        # Works yay
        
        # Testing semi-vectorised accel step
        # return self.semivector_accel_step(self.mPos_Bodies, self.vMass_Bodies, self.vRadius_bodies, self.fGravity_Constant)
        # Works yay
        print('testing')

nbod = N_Body_Solver(100, 100, 10, 1)
nbod.prepare()
Accels = nbod.test()


print('EOF')
        



                
