
from math import ceil

import numpy as np
import cupy as cp

from particles import Particles


class CUSPH():
    def __init__(self, cfg):
        self.cfg = cfg
        self.particles = Particles(cfg)
        self.particles.cuda()
        self.irho = cp.empty(len(self.particles))
        self.cuda()

    def cuda(self):
        self.base = cp.float32(self.particles.n_cells)
        self.smoothlen = cp.float32(self.cfg.smoothlen)
        self.c0 = cp.int32(self.particles.c0)
        self.c1 = cp.int32(self.particles.c1)
        self.c2 = cp.int32(self.particles.c2)
        self.dcell = cp.int32(self.cfg.dcell)
        self.coef_density = cp.float32(self.cfg.coef_density)
        self.coef_pressure = cp.float32(self.cfg.coef_pressure)
        self.coef_viscosity = cp.float32(self.cfg.coef_viscosity)
        self.coef_rhop0 = cp.float32(self.cfg.rhop0)
        self.rhop0 = cp.float32(self.cfg.rhop0)
        self.gamma = cp.float32(self.cfg.gamma)
        self.B = cp.float32(self.cfg.B)
        self.mu = cp.float32(self.cfg.mu)
        self.gravity = cp.array(self.cfg.gravity, dtype=np.float32)


    def print(self, text):
        if self.cfg.verbose > 0:
            print(text)


    def compute_step(self):
        self.particles.set_cell()

        s_inputs = []
        v_inputs = []
        # hash
        s_hash = ['int32 hashid', 'raw int32 cellstart', 'raw int32 cellend', 'raw int32 sortids']
        v_hash = [self.particles.hashids, self.particles.cellstart, self.particles.cellend, self.particles.sortids]
        
        # cell
        s_cell = ['int32 base, int32 c0, int32 c1, int32 c2, int32 dcell']
        v_cell = [self.base, self.c0, self.c1, self.c2, self.dcell]

        # val for density and pressure
        s_input1 = ['float32 smoothlen', 'raw float32 pos', 'float32 coef_density', 'float32 rhop0', 'float32 gamma', 'float32 B']
        v_input1 = [self.smoothlen, self.particles.pos, self.coef_density, self.rhop0, self.gamma, self.B]

        s_output1 = ['float32 density', 'float32 press']
        v_output1 = [self.particles.density, self.particles.press]

        cp.ElementwiseKernel(
            ', '.join(s_hash + s_cell + s_input1),
            ', '.join(s_output1),
            '''
            // hashid to cell coordinate
            int hd0, hd1, hd2;
            hd0 = hashid;
            hd2 = hd0  / (base * base);
            hd0 = hd0 - hd2 * base * base;
            hd1 = hd0 / base;
            hd0 = hd0 - hd1 * base;

            //
            float h = smoothlen;
            float h_sq = pow(smoothlen, (float)2.0);

            int neighbor_id;
            int startidx;
            int endidx;
            int idx0[2];
            int idx1[2];
            float r_sq;

            density = 0.0;
            press = 0.0;
            idx0[0] = i;
            for(int i0=max(hd0-dcell, 0); i0<=min(hd0+dcell, c0); i0++){
                for(int i1=max(hd1-dcell, 0); i1<=min(hd1+dcell, c1); i1++){
                    for(int i2=max(hd2-dcell, 0); i2<=min(hd2+dcell, c2); i2++){
                        neighbor_id = i0 + i1 * base + i2 * base * base;
                        startidx = cellstart[neighbor_id];
                        endidx = cellend[neighbor_id];
                        for(int k=startidx; k<endidx; k++){
                            if(i==k) continue;
                            idx1[0] = sortids[k];
                            r_sq = 0.0;
                            for(int d=0;d<3;d++){
                                idx0[1] = d;
                                idx1[1] = d;
                                r_sq += pow(pos[idx0] - pos[idx1], (float)2.0);
                            }
                            if(r_sq < h_sq){
                                 density += pow(h_sq - r_sq, (float)3.0);
                            }
                        }
                    }
                }
            }
            density = density * coef_density;
            press = B * (pow(density * coef_density / rhop0, gamma) - 1.0);
            if(press < 0.0) press = 0.0;
            ''',
            'calc_density'
            )(*v_hash, *v_cell, *v_input1, *v_output1)

        s_output1 = ['raw float32 density', 'raw float32 press']

        s_input2 = ['float32 smoothlen', 'raw float32 pos', 'raw float32 vel', 'float32 coef_pressure', 'float32 coef_viscosity', 'float32 mu']
        v_input2 = [self.smoothlen, self.particles.pos, self.particles.vel, self.coef_pressure, self.coef_viscosity, self.mu]

        s_output2 = ['raw float32 accel']
        v_output2 = [self.particles.accel]

        cp.ElementwiseKernel(
            ', '.join(s_hash + s_cell + s_output1 + s_input2),
            ', '.join(s_output2),
            '''
            // hashid to cell coordinate
            int hd0, hd1, hd2;
            hd0 = hashid;
            hd2 = hd0  / (base * base);
            hd0 = hd0 - hd2 * base * base;
            hd1 = hd0 / base;
            hd0 = hd0 - hd1 * base;

            //
            float h = smoothlen;
            float h_sq = pow(smoothlen, (float)2.0);

            int neighbor_id;
            int startidx;
            int endidx;
            int idx0[2];
            int idx1[2];
            float r_sq;
            float dr[3];
            int j;

            for(int d=0;d<3;d++){
                accel[i*3+d] = 0.0;
            }
            idx0[0] = i;
            for(int i0=max(hd0-dcell, 0); i0<=min(hd0+dcell, c0); i0++){
                for(int i1=max(hd1-dcell, 0); i1<=min(hd1+dcell, c1); i1++){
                    for(int i2=max(hd2-dcell, 0); i2<=min(hd2+dcell, c2); i2++){
                        neighbor_id = i0 + i1 * base + i2 * base * base;
                        startidx = cellstart[neighbor_id];
                        endidx = cellend[neighbor_id];
                        for(int k=startidx; k<endidx; k++){
                            if(i==k) continue;
                            j = sortids[k];
                            idx1[0] = j;
                            r_sq = 0.0;
                            for(int d=0;d<3;d++){
                                idx0[1] = d;
                                idx1[1] = d;
                                dr[d] = pos[idx0] - pos[idx1];
                                r_sq += pow(pos[idx0] - pos[idx1], (float)2.0);
                            }
                            float r = sqrtf(r_sq);
                            float c = h - r;
                            if(r_sq < h_sq){
                                float pterm = coef_pressure * (press[i] - press[j]) / (float)2.0 * pow(c, (float)2.0) / r;
                                float vterm = coef_viscosity * mu * c;
                                for(int d=0;d<3;d++){
                                    idx0[1] = d;
                                    idx1[1] = d;
                                    accel[i*3+d] += (pterm * dr[d] + vterm * (vel[j] - vel[i])) / density[i] / density[j];
                                }
                            }
                        }
                    }
                }
            }
            ''',
            'calc_accel'
            )(*v_hash, *v_cell, *v_output1, *v_input2, *v_output2, size=len(self.particles))


    def integrate(self):
        self.print("update_particle")
        accel = self.particles.accel.copy()

        speed = cp.sum(cp.square(accel), axis=1, keepdims=True)
        condition = cp.broadcast_to(speed > self.cfg.limit**2, accel.shape)
        accel = cp.where(condition, accel*self.cfg.limit/np.sqrt(speed), accel)

        h = self.cfg.smoothlen
        # 壁境界
        scale = 0.004
        xlim = [0.0, 20.0 * scale]
        ylim = [0.0, 50.0 * scale]
        zlim = [-10.0 * scale, 10.0 * scale]

        for i, lim in zip([0,1,2], [xlim, ylim, zlim]):
            diff = 2.0 * self.cfg.radius - (self.particles.pos[:,i] - lim[0])
            adj = self.cfg.wall * diff - self.cfg.damp * self.particles.vel[:,i]
            accel[:,i] += cp.where(diff > 0, adj, 0.0)

            diff = 2.0 * self.cfg.radius - (lim[1] - self.particles.pos[:,i])
            adj = self.cfg.wall * diff + self.cfg.damp * self.particles.vel[:,i]
            accel[:,i] -= cp.where(diff > 0, adj, 0.0)

        # 重力の加算
        accel += self.gravity

        self.particles.vel += self.cfg.time_step * accel
        self.print(f"v: {self.particles.vel[0]}")
        self.particles.pos += self.cfg.time_step * self.particles.vel


