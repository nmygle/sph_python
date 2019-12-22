import numpy as np
from numba import jitclass
from numba import types, typeof
from numba import int32, float32, int64, float64
from numba import deferred_type
from numba.typed import List
from numba import typed

spec = [
    ("id", int64),
    ("position", float64[:]),
    ("velocity", float64[:]),
    ("neighbor", int64[:])
]
@jitclass(spec)
class Particle():
    def __init__(self, idx, position, velocity):
        self.position = position
        self.velocity = velocity
        self.neighbor = np.empty(0, dtype=np.int64)

    def append(self, val):
        self.neighbor = np.append(self.neighbor, val)

particle_type = deferred_type()
particle_type.define(Particle.class_type.instance_type)

pt_cls = Particle.class_type.instance_type
list_of_pt = types.ListType(pt_cls)
#@jitclass([("particles", types.ListType(particle_type))])
@jitclass([("particles", types.ListType(Particle.class_type.instance_type))])
class Particles():
    def __init__(self):
        self.particles = List.empty_list(pt_cls)
        scale = 0.004
        dx = (0.00020543/600) ** (1/3) / scale * 0.95
        x_range = [0.0 + dx, 10.0 - dx]
        y_range = [0.0 + dx, 20.0 - dx]
        z_range = [-10.0 + dx, 10.0 - dx]
        count = 0
        for iz in np.arange(z_range[0], z_range[1], dx):
            for ix in np.arange(x_range[0], x_range[1], dx):
                for iy in np.arange(y_range[0], y_range[1], dx):
                    px = ix * scale
                    py = iy * scale
                    pz = iz * scale
                    pos = np.array([px, py, pz], dtype=np.float64)
                    vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    self.particles.append(Particle(count, pos, vel))
                    count += 1
        #particle = Particle(count, pos, vel)
        #particles.append(particle)
        #self.particles = particles


    def __len__(self):
        return len(self.particles)

if __name__ == "__main__":
    particles = Particles()
    print(particles.particles[0].position)
    assert False
    count = 0
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])
    particle = Particle(count, pos, vel)
    particles = List()
    particles.append(particle)
    print(typeof(particles))
    assert False

