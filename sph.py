
import numpy as np
from numba import jit, prange

from tqdm import tqdm


class Particles():
    def __init__(self):
        init_pos = []
        init_vel = []
        dx = 0.001
        x_range = [-1.0, -0.9]
        y_range = [-1.0, -0.9]
        z_range = [0.0, dx]
        for iz in np.arange(z_range[0], z_range[1], dx):
            for ix in np.arange(x_range[0], x_range[1], dx):
                for iy in np.arange(y_range[0], y_range[1], dx):
                    px = ix
                    py = iy
                    pz = iz
                    px += -dx * 0.1 + np.random.normal(0.0, dx*0.01)
                    py += -dx * 0.1 + np.random.normal(0.0, dx*0.01)
                    pz += -dx * 0.1 + np.random.normal(0.0, dx*0.01)
                    init_pos.append([px, py, pz])
                    init_vel.append([0.0, 0.0, 0.0])
        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)

    def __len__(self):
        return len(self.pos)


@jit('Tuple((f8[:], f8[:]))(f8[:,:], f8, f8, f8, f8, f8)', nopython=True, parallel=True)
def calc_density_pressure(pos, smoothlen, coef_density, rhop0, gamma, B):
    h_sq = smoothlen * smoothlen
    particles_irho = np.empty(len(pos))
    particles_press = np.empty(len(pos))
    for i in prange(len(pos)):
        rho = 0.0
        for j in range(len(pos)):
            if i == j:
                continue
            r_sq = np.sum(np.square(pos[i] - pos[j]))
            if h_sq > r_sq:
                rho += (h_sq - r_sq) ** 3.0
        particles_irho[i] = 1.0 / (rho * coef_density)
        particles_press[i] = B * np.power(1.0 / particles_irho[i] / rhop0, gamma) - 1.0
        if particles_press[i] < 0.0:
            particles_press[i] = 0.0

    return particles_irho, particles_press



@jit('f8[:,:](f8[:,:], f8[:,:], f8[:], f8[:], f8, f8, f8, f8)', nopython=True, parallel=True)
def calc_accel(pos, vel, press, idensity, smoothlen, coef_pressure, coef_viscosity, mu):
    h = smoothlen
    accel = np.zeros((len(pos), 3))
    for i in prange(len(pos)):
        for j in range(len(pos)):
            if i == j:
                continue
            #dr = pos[i] - pos[j]
            dr = pos[j] - pos[i]
            r = np.sqrt(np.sum(np.square(dr)))
            if h > r:
                c = h - r
                pterm = coef_pressure * (press[i] + press[j]) / 2 * c ** 2 / r
                vterm = coef_viscosity * mu * c
                fcurr = pterm * dr + vterm * (vel[j] - vel[i])
                fcurr *= idensity[i] * idensity[j]
                accel[i] += fcurr
    return accel


class FluidSolver():
    def __init__(self, cfg):
        self.cfg = cfg
        self.particles = Particles()

    def print(self, text):
        if self.cfg.verbose > 0:
            print(text)

    def compute_step(self):
        h = self.cfg.smoothlen
        h_sq = self.cfg.smoothlen ** 2
        n_particle = len(self.particles)

        self.print("calc density and press")
        idensity, press = calc_density_pressure(self.particles.pos, self.cfg.smoothlen, self.cfg.coef_density, self.cfg.rhop0, self.cfg.gamma, self.cfg.B)

        self.print("calc press and viscosity")
        accel = calc_accel(self.particles.pos, self.particles.vel, idensity, press, self.cfg.smoothlen, self.cfg.coef_pressure, self.cfg.coef_viscosity, self.cfg.mu)

        self.particles.accel = accel


    def integrate(self):
        self.print("update_particle")
        accel = self.particles.accel

        h = self.cfg.smoothlen
        # 壁境界
        xlim = [-1.0, 1.0]
        ylim = [-1.0, 1.0]
        zlim = [-1.0, 1.0]
        
        for i, lim in zip([0,1,2], [xlim, ylim, zlim]):
            diff = 2.0 * self.cfg.radius - (self.particles.pos[:,i] - lim[0])
            adj = self.cfg.wall * diff - self.cfg.damp * self.particles.vel[:,i]
            accel[:,i] += np.where(diff > 0, adj, 0.0)

            diff = 2.0 * self.cfg.radius - (lim[1] - self.particles.pos[:,i])
            adj = self.cfg.wall * diff + self.cfg.damp * self.particles.vel[:,i]
            accel[:,i] -= np.where(diff > 0, adj, 0.0)

        # 重力の加算
        accel += self.cfg.gravity

        self.particles.vel += self.cfg.time_step * accel
        self.print(f"v: {self.particles.vel[0]}")
        self.particles.pos += self.cfg.time_step * self.particles.vel


    def run(self):
        out_dir = "results"
        for t in tqdm(range(self.cfg.n_time), ncols=45):
        #for t in range(self.cfg.n_time):
            #print("-------------", t, "------------")
            self.compute_step()
            self.integrate()
            save(self.particles.pos, f"{out_dir}/{t}.p")


class Params():
    def __init__(self):
        self.verbose = 0
        self.n_time = 100

        self.eps = 1.0e-12
        self.time_step = 0.005
        self.smoothlen = 0.012

        self.radius = 0.004
        
        self.wall = 10000.0
        self.damp = 256.0

        particle_mass = 0.00020543
        self.particle_mass = particle_mass
        # カーネル係数
        self.coef_density = particle_mass * 4 / (np.pi * np.power(self.smoothlen, 8))
        self.coef_pressure = particle_mass * (-30.0) / (np.pi * np.power(self.smoothlen, 5))
        self.coef_viscosity = particle_mass * (20/3) / (np.pi * np.power(self.smoothlen, 5))
        # for pressure
        # reference density []
        self.rhop0 = 1000.0
        self.gamma = 7
        self.hswl = 0
        # 圧力項係数
        if self.hswl == 0:
            self.B = 200
        else:
            self.coefsound = 20
            self.cs = self.coefsound * np.sqrt(9.8 * self.hswl)
            self.B = self.cs ** 2 * self.rhop0 / self.gamma
        # 粘性
        self.mu = 0.1
        # 重力
        self.gravity = np.array([0.0, -9.8, 0.0])


import pickle
def save(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)

def mkmove(n_time):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    ims = []
    for t in range(n_time):
        with open(f"results/{t}.p", "rb") as file:
            data = pickle.load(file)
        im = plt.plot(data[:,0], data[:,1], ".", c="blue")
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("/mnt/c/space/tm.mp4", writer="ffmpeg")

def main():
    #import os
    #os.makedirs(out_dir, exist_ok=True)

    cfg = Params()
    fs = FluidSolver(cfg)
    print(f'n_particles:{len(fs.particles)}')
    fs.run()
    mkmove(cfg.n_time)

if __name__ == "__main__":
    main()

