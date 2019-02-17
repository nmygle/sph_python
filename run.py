
import os, time
import pickle

from tqdm import tqdm

import cupy as cp

from config import Params
from solvers import CUSPH


def save(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def mkmove(n_time, outdir):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    ims = []
    for t in range(n_time):
        with open(f"{outdir}/{t}.p", "rb") as file:
            data = pickle.load(file)
        im = plt.plot(data[:,0], data[:,1], ".", c="blue")
        #lim = 0.1
        #plt.xlim(-lim, lim)
        #plt.ylim(-lim, lim)
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save(f"{outdir}/movie.mp4", writer="ffmpeg")


def main():
    cfg = Params()
    sph = CUSPH(cfg)

    print(f'n_particles:{len(sph.particles)}')

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    n_time = 6
    flg_save = False
    start = time.time()
    #for t in tqdm(range(n_time), ncols=45):
    for t in range(n_time):
        print("-------------", t, "------------")
        sph.compute_step()
        sph.integrate()
        if flg_save:
            save(cp.asnumpy(sph.particles.pos), f"{out_dir}/{t}.p")
    print("finish:", time.time() - start)

    if flg_save:
        mkmove(n_time, out_dir)


if __name__ == "__main__":
    main()


