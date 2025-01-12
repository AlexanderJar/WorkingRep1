import sys
import numpy as np
import pygimli as pg
from fpinv import FourPhaseModel


# Time-lapse scenario test case
# Iterate over time steps

ntimes = 3

for timestep in range(1,ntimes+1):

    mesh = pg.load("mesh_TL.bms")

    pd = pg.load("paraDomain_TL.bms")
    resinv = np.loadtxt("res_conventional_TL_t%s_lam100.dat" % timestep)
    vest = np.loadtxt("vel_conventional_TL_t%s.dat" % timestep)
    phi = 0.3  # Porosity assumed to calculate fi, fa, fw with 4PM

    # Save some stuff
    fpm = FourPhaseModel(phi=phi)
    fae, fie, fwe, maske = fpm.all(resinv, vest)
    print(np.min(fwe), np.max(fwe))
    np.savez("conventional_TL_t%s_lam100.npz" % timestep, vel=np.array(vest),
            rho=np.array(resinv), fa=fae, fi=fie, fw=fwe, mask=maske)
