import numpy as np

class Params():
    def __init__(self):
        self.verbose = 0
        self.n_time = 200
        self.limit = 200.0

        self.time_step = 0.004
        self.smoothlen = 0.012

        self.radius = 0.004
        self.wall = 10000.0
        self.damp = 256.0

        # 近傍計算の分割数
        self.dcell = 2

        particle_mass = 0.00020543
        self.particle_mass = particle_mass
        # カーネル係数
        self.coef_density = particle_mass * 315.0 / (64.0 * np.pi * np.power(self.smoothlen, 9))
        self.coef_pressure = particle_mass * (-45.0) / (np.pi * np.power(self.smoothlen, 6))
        self.coef_viscosity = particle_mass * 45.0 / (np.pi * np.power(self.smoothlen, 6))
        # for pressure
        # reference density []
        self.rhop0 = 600.0
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

