
from math import ceil

import numpy as np
import cupy as cp


class Particles():
    def __init__(self, cfg):
        self.cfg = cfg
        # set initial state
        init_pos = []
        init_vel = []
        scale = 0.004
        dx = (0.00020543/600) ** (1/3) / scale * 0.95
        x_range = [0.0 + dx, 10.0 - dx]
        y_range = [0.0 + dx, 20.0 - dx]
        z_range = [-10.0 + dx, 10.0 - dx]
        for iz in np.arange(z_range[0], z_range[1], dx):
            for ix in np.arange(x_range[0], x_range[1], dx):
                for iy in np.arange(y_range[0], y_range[1], dx):
                    px = ix * scale
                    py = iy * scale
                    pz = iz * scale
                    init_pos.append([px, py, pz])
                    init_vel.append([0.0, 0.0, 0.0])
        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)

        self.x_min = np.min(self.pos, axis=0) - (5 * scale)
        self.x_max = np.max(self.pos, axis=0) + (5 * scale)

        self.dx = self.cfg.smoothlen / self.cfg.dcell;
        n_cells_xyz = (self.x_max - self.x_min - 1.0e-11) // self.dx + 1
        self.c0 = n_cells_xyz[0]
        self.c1 = n_cells_xyz[1]
        self.c2 = n_cells_xyz[2]

        self.n_cells = ceil(np.max(n_cells_xyz))
        self.hashids = np.zeros(len(self), dtype=np.float32)
        self.cellstart = np.empty(self.n_cells ** 3)
        self.cellend = np.empty(self.n_cells ** 3)
        self.cellstart.fill(-1)
        self.cellend.fill(-1)

        self.density = np.zeros(len(self), dtype=np.float32)
        self.press = np.zeros(len(self), dtype=np.float32)
        self.accel = np.zeros([len(self),3], dtype=np.float32)


    def encode(self, idxs):
        return int(sum([idx * self.n_cells ** k for k, idx in enumerate(idxs)]))


    def set_cell(self):
        # 初期化
        self.hashids.fill(0)
        self.cellstart.fill(-1)
        self.cellend.fill(-1)

        # 粒子にグリッドインデックスを登録
        for i in range(len(self)):
            idx = (self.pos[i] - self.x_min) // self.dx
            # debug
            try:
                self.hashids[i] = self.encode(idx)
            except:
                print("error:", self.pos[i])
                assert False

        # sort
        self.sortids = self.hashids.argsort().astype(self.hashids.dtype)

        # 開始点、終了点の登録
        pre = self.hashids[self.sortids[0]]
        self.cellstart[pre] = 0
        for i in range(1, len(self)):
            if pre == self.hashids[self.sortids[i]]:
                pass
            else:
                self.cellend[pre] = i-1
                pre = self.hashids[self.sortids[i]]
                self.cellstart[pre] = i
        self.cellend[pre] = i


    def cuda(self):
        self.pos = cp.array(self.pos, dtype=np.float32)
        self.vel = cp.array(self.vel, dtype=np.float32)
        self.density = cp.array(self.density, dtype=np.float32)
        self.press = cp.array(self.press, dtype=np.float32)
        self.accel = cp.array(self.accel, dtype=np.float32)
        self.hashids = cp.array(self.hashids, dtype=np.int32)
        self.cellstart = cp.array(self.cellstart, dtype=np.int32)
        self.cellend = cp.array(self.cellend, dtype=np.int32)
        self.x_min = cp.array(self.x_min, dtype=np.float32)
        self.x_max = cp.array(self.x_max, dtype=np.float32)


    def numpy(self):
        self.pos = cp.asnumpy(self.pos)
        self.vel = cp.asnumpy(self.vel)
        self.density = cp.asnumpy(self.density)
        self.press = cp.asnumpy(self.press)
        self.accel = cp.asnumpy(self.accel)


    def __len__(self):
        return self.pos.shape[0]


