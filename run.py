
from config import Params
from solvers import CUSPH


def main():
    cfg = Params()
    sph = CUSPH(cfg)

    sph.compute_step()

    #print(f'n_particles:{len(fs.particles)}')
    #fs.run()
    #mkmove(cfg.n_time)

if __name__ == "__main__":
    main()

