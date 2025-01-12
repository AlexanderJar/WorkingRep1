import sys
import numpy as np
import pygimli as pg
from fpinv import FourPhaseModel


# Time-lapse scenario test case
# Iterate over time steps

ntimes = 3

for timestep in range(1,ntimes+1):

    mesh = pg.load("mesh_TL.bms")

    if len(sys.argv) > 1:
        pd = pg.load("paraDomain_TL_2.bms")
        resinv = np.loadtxt("res_conventional_TL_2_t%s_lam100.dat" % timestep)
        vest = np.loadtxt("vel_conventional_TL_2_t%s.dat" % timestep)
        scenario = "Fig2"
        pg.boxprint(scenario)
        phi = 0.3  # Porosity assumed to calculate fi, fa, fw with 4PM
    else:
        pd = pg.load("paraDomain_TL_1.bms")
        resinv = np.loadtxt("res_conventional_TL_1_t%s.dat" % timestep)
        vest = np.loadtxt("vel_conventional_TL_1_t%s.dat" % timestep)
        scenario = "Fig1"
        pg.boxprint(scenario)
        frtrue = np.load("true_model_TL.npz")["fr"]

        phi = []
        for cell in pd.cells():
            idx = mesh.findCell(cell.center()).id()
            phi.append(1 - frtrue[idx])
        phi = np.array(phi)

    # Save some stuff
    fpm = FourPhaseModel(phi=phi)
    fae, fie, fwe, maske = fpm.all(resinv, vest)
    print(np.min(fwe), np.max(fwe))
    np.savez("conventional_TL_%s_t%s_lam100.npz" % (scenario, timestep), vel=np.array(vest),
            rho=np.array(resinv), fa=fae, fi=fie, fw=fwe, mask=maske)
