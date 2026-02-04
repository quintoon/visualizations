from orbitize.hipparcos import PMPlx_Motion
from orbitize.kepler import calc_orbit
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import glob
import os


sma = 1.5
ecc = 0.5
inc = 0
aop = 0
pan = 0
tau = 0
mtot = 1
mp = 0.2
per = np.sqrt(sma**3 / mtot)

epochs_mjd = Time(np.linspace(2010, 2010 + 4 * per, int(1e3)), format="decimalyear").mjd

myAstromModel = PMPlx_Motion(epochs_mjd, 10, 30, alphadec0_epoch=2010)

plx = 20  # [mas]
pm_ra = 40  # [mas/yr]
pm_dec = 40  # [mas/yr]
alpha_H0 = 0
delta_H0 = 0

param_idx = {"plx": 0, "pm_ra": 1, "pm_dec": 2, "alpha0": 3, "delta0": 4}
ra, dec = myAstromModel.compute_astrometric_model(
    [plx, pm_ra, pm_dec, alpha_H0, delta_H0], param_idx
)


ra_kep, dec_kep, _ = calc_orbit(epochs_mjd, sma, ecc, inc, aop, pan, tau, plx, mtot)
ra_star = ra_kep * mp / mtot
dec_star = dec_kep * mp / mtot

ra_pl = -ra_kep
dec_pl = -dec_kep

plot_every = 10

if not os.path.exists("astrom_gif"):
    os.mkdir("astrom_gif")
os.system("rm astrom_gif/*png")

for i, t in enumerate(epochs_mjd[::plot_every]):

    plot_index = i * plot_every

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(ra_star, dec_star, color="rebeccapurple")
    ax[0].plot(
        ra_pl,
        dec_pl,
        color="rebeccapurple",
        alpha=0.2,
        ls="--",
    )
    ax[1].plot(ra, dec, ls=":", color="grey", label="star motion")
    ax[1].plot(
        ra + ra_star,
        dec + dec_star,
        ls="-",
        color="rebeccapurple",
        label="star+planet motion",
    )
    ax[1].set_xlabel("$\\Delta$R.A.cos($\\delta_0$) [mas]")
    ax[1].set_ylabel("$\\Delta$decl. [mas]")

    for a in ax.flatten():
        a.set_aspect("equal")
    ax[0].scatter([0], [0], marker="x", color="k", label="barycenter")

    ax[0].scatter(
        ra_star[plot_index],
        dec_star[plot_index],
        color="rebeccapurple",
        marker="*",
        s=75,
        label="star orbit",
    )
    ax[0].scatter(
        ra_pl[plot_index],
        dec_pl[plot_index],
        color="rebeccapurple",
        zorder=10,
        label="planet orbit",
    )
    ax[1].scatter(
        ra[plot_index] + ra_star[plot_index],
        dec[plot_index] + dec_star[plot_index],
        color="rebeccapurple",
        marker="*",
        s=75,
        zorder=10,
    )

    l = ax[0].legend(loc="upper left")
    l.set_zorder(20)
    ax[1].legend(loc="upper left")
    ax[0].set_xlabel("$\\Delta$ R.A. [mas]")
    ax[0].set_ylabel("$\\Delta$ decl. [mas]")

    plt.tight_layout()
    plt.savefig(f"astrom_gif/astrom{i}.png", dpi=250)
    plt.close()


ims = [
    imageio.imread(f"astrom_gif/astrom{i}.png")
    for i in range(len(epochs_mjd[::plot_every]))
]
imageio.mimwrite("astrom_gif.gif", ims, loop=1000)
