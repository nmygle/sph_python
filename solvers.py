
import time
from math import ceil

import numpy as np
import cupy as cp


class Particles():
    def __init__(self):
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
        self.xmin = np.min(self.pos, axis=0)
        self.xmax = np.max(self.pos, axis=0)

        self.dx = self.cfg.smoothlen / 2
        n_cells_xyz = (self.x_max - self.x_min - 1.0e-11) // self.dx + 1

        self.n_cells = ceil(np.max(n_cells_xyz))

        self.hashids = np.zeros(len(self), dtype=int)
        self.cellstart = np.empty(n_cells ** 3).fill(-1)
        self.cellend = np.empty(n_cells ** 3).fll(-1)


    def encode(self, idxs):
        return int([idx * self.n_cells ** k for k, idx in enumerate(idxs)])


    def add_hashid(self):
        # 初期化
        self.hashids.fill(0)
        self.cellstart.fill(-1)
        self.cellend.fill(-1)

        # 粒子にグリッドインデックスを登録
        for i in range(len(self)):
            idx = (self.pos[i] - self.x_min) // self.dx
            self.hashids[i] = self.encode(idx)

        # sort
        self.sortids = hashids.argsort()

        # 開始点、終了点の登録
        pre = self.hashids[self.sortids[0]]
        self.cellstart[pre] = 0
        for i in range(1, len(self)):
            if pre == self.hashids[self.sortids[i]]:
                pass
            else:
                self.cellend[pre] = i-1
                self.cellstart[pre] = i
        self.cellend[pre] = i


    def cuda(self):
        self.pos = cp.array(self.pos, dtype=np.float32)
        self.vel = cp.array(self.vel, dtype=np.float32)
        self.hashids = cp.array(hashids, dtype=np.int32)
        cellstart = cp.array(cellstart, dtype=np.int32)
        cellend = cp.array(cellend, dtype=np.int32)
        sortids = cp.array(sortids)


    def numpy(self):
        self.pos = cp.asnumpy(self.pos)
        self.vel = cp.asnumpy(self.vel)


    def __len__(self):
        return self.pos.shape[0]


class CUSPH():
    def __init__(self, cfg):
        self.cfg = cfg
        self.particles = Particles()


    def cuda(self):
        self.particles.cuda()


    def print(self, text):
        if self.cfg.verbose > 0:
            print(text)


    def compute_step(self):
        h = self.cfg.smoothlen
        h_sq = self.cfg.smoothlen ** 2
        n_particle = len(self.particles)

        cp.ElementwiseKernel(
            'int32 hashid, raw T pos, raw int32 cellstart, raw int32 cellend, raw int32 sortids, int32 base, float32 rc, int32 c0, int32 c1, int32 c2',
            'T dense',
            '''
            // hashid to cell coordinate
            int hs0, hd1, hd2;
            hd0 = hashid;
            hd2 = hd0  / (base * base);
            hd0 = hd0 - hd2 * base * base;
            hd1 = hd0 / base;
            hd0 = hd0 - hd1 * base;

            //printf("[%d,%d,%d],", hd0, hd1, hd2);

            int neighbor_id;
            int startidx;
            int endidx;
            int idx0[2];
            int idx1[2];
            float dist;
            for(int i0=max(hd0-2, 0); i0<min(hd0+2, c0); i0++){
                for(int i0=max(hd0-2, 0); i0<min(hd0+2, c0); i0++){
                    for(int i0=max(hd0-2, 0); i0<min(hd0+2, c0); i0++){
                        neighbor_id = i0 + i1 * base + i2 * base * base;
                        startidx = cellstart[neighbor_id];
                        endidx = cellend[neighbor_id];
                        idx0[0] = hashid;
                        for(int k=startidx; k<endidx; k++){
                            if(i==k) continue;
                            idx1[0] = sortids[k];
                            dist = 0.0;
                            for(d=0;d<3;d++){
                                idx0[1] = d;
                                idx1[1] = d;
                                dist += pow(pos[idx0] - pos[idx1], (float)2.0);
                            }
                            if(dist < pow(rc, (float)2.0)){
                                dense += exp(-dist/rc);
                            }

                        }
                    }
                }
            }
            ''',
            'calc_neighbor'
            )(hashids, pos, cellstart, cellend, sortids, n_cells, rc, c0, c1, c2, dense)

        self.particles.accel = accel

        #self.print("calc density and press")
        #idensity, press = calc_density_pressure(self.particles.pos, self.cfg.smoothlen, self.cfg.coef_density, self.cfg.rhop0, self.cfg.gamma, self.cfg.B)

        #self.print("calc press and viscosity")
        #accel = calc_accel(self.particles.pos, self.particles.vel, idensity, press, self.cfg.smoothlen, self.cfg.coef_pressure, self.cfg.coef_viscosity, self.cfg.mu)
        #start = time.time()


    def integrate(self):
        self.print("update_particle")
        accel = self.particles.accel

        speed = np.sum(np.square(accel), axis=1, keepdims=True)
        condition = np.broadcast_to(speed > self.cfg.limit**2, accel.shape)
        accel = np.where(condition, accel*self.cfg.limit/np.sqrt(speed), accel)

        h = self.cfg.smoothlen
        # 壁境界
        scale = 0.004
        xlim = [0.0, 20.0 * scale]
        ylim = [0.0, 50.0 * scale]
        zlim = [-10.0 * scale, 10.0 * scale]
        
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
        os.makedirs(out_dir, exist_ok=True)
        #for t in tqdm(range(self.cfg.n_time), ncols=45):
        for t in range(self.cfg.n_time):
            print("-------------", t, "------------")
            self.compute_step()
            self.integrate()
            save(self.particles.pos, f"{out_dir}/{t}.p")



