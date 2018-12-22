
import numpy as np
from collections import namedtuple

from tqdm import tqdm


class Particles():
    def __init__(self):
        init_pos = []
        init_vel = []
        dx = 0.01
        for ix in np.arange(0.0, 1.0, dx):
            for iy in np.arange(0.0, 1.0, dx):
                for iz in np.arange(0.0, 1.0, dx):
                    init_pos.append([ix, iy, iz])
                    init_vel.append([0.0, 0.0, 0.0])
        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)

    def __len__(self):
        return len(self.pos)


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

        # 距離計算
        self.print("calc_distance")
        pos_vec = np.expand_dims(self.particles.pos, 0)
        pos_n = np.repeat(pos_vec, n_particle, axis=0)
        pos_p = np.repeat(pos_vec, n_particle, axis=1).reshape(n_particle, n_particle, 3)
        r_sq_matrix = np.linalg.norm(pos_n - pos_p, axis=2)

        mask_same = r_sq_matrix > 0

        # 密度計算
        self.print("calc_rho")
        rho_mat = self.cfg.coef_density * (h_sq - r_sq_matrix) ** 3
        mask_cutoff = rho_mat > 0
        mask = mask_same & mask_cutoff
        rho_mat = np.where(mask, rho_mat, 0.0)
        self.particles.rho = np.sum(rho_mat, axis=1, keepdims=True) # [n_particle, 1]

        # 粒子毎の圧力計算
        # Tait equation
        self.print("calc_press_echo_particle")
        self.particles.pressure = self.cfg.B * \
                np.maximum(np.power(self.particles.rho / self.cfg.rhop0, self.cfg.gamma) - 1, 0) # [n_particle, 1]

        # 圧力項
        self.print("calc_press")
        press_vec = self.particles.pressure.reshape(1, -1)
        press_n = np.repeat(press_vec, n_particle, axis=0)
        press_p = np.repeat(press_vec, n_particle, axis=1).reshape(n_particle, n_particle)
        avg_pressure = (press_n + press_p) / 2.0
        r = np.sqrt(r_sq_matrix)
        diff = (pos_n - pos_p)

        press_mat = self.cfg.coef_pressure * avg_pressure / (self.particles.rho + self.cfg.eps) \
                    * (h - r)**2 / (r + self.cfg.eps)
        press_mat = np.expand_dims(press_mat, -1) / (diff + self.cfg.eps)
        mask3d = np.broadcast_to(np.expand_dims(mask, -1), [n_particle, n_particle, 3])
        press_f = np.sum(np.where(mask3d,  press_mat, 0.0), axis=1) # [n_particle, 3]

        # 粘性項
        self.print("calc_viscosity")
        velocity_vec = np.expand_dims(self.particles.vel, 0)
        v_n = np.repeat(velocity_vec, n_particle, axis=0)
        v_p = np.repeat(velocity_vec, n_particle, axis=1).reshape(n_particle, n_particle, 3)
        visco_mat = self.cfg.coef_viscosity * (h - r) / (self.particles.rho + self.cfg.eps)
        visco_mat = np.expand_dims(visco_mat, -1) * (v_n - v_p)
        visco_f = np.sum(np.where(mask3d,  visco_mat, 0.0), axis=1) # [n_particle, 3]

        force = press_f + self.cfg.viscosity * visco_f
        print(force)
        self.particles.acceleration = force / (self.particles.rho + self.cfg.eps)


    def integrate(self):
        self.print("update_particle")
        acceleration = self.particles.acceleration

        # 重力の加算
        acceleration += self.cfg.gravity
        print(acceleration)
        assert False

        self.particles.vel += self.cfg.time_step * acceleration
        self.particles.pos += self.cfg.time_step * self.particles.vel


    def run(self):
        for t in tqdm(range(100), ncols=45):
            self.compute_step()
            self.integrate()


class Params():
    def __init__(self):
        self.verbose = 1
        self.eps = 1.0e-7
        self.time_step = 0.01
        self.smoothlen = 0.012
        
        particle_mass = 0.0002
        # カーネル係数
        self.coef_density = particle_mass * 4 / (np.pi * np.power(self.smoothlen, 8))
        self.coef_pressure = particle_mass * (-30.0) / (np.pi * np.power(self.smoothlen, 5))
        self.coef_viscosity = particle_mass * (20/3) / (np.pi * np.power(self.smoothlen, 5))
        # for pressure
        self.mu = 1.0
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
        self.viscosity = 0.1
        # 重力
        self.gravity = np.array([0.0, 0.0, -9.8])


import pickle
def save(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)

def main():
    import os
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    cfg = Params()
    fs = FluidSolver(cfg)
    save(fs.particles.pos, f"{out_dir}/0.p")
    fs.run()
    save(fs.particles.pos, f"{out_dir}/100.p")

if __name__ == "__main__":
    main()

