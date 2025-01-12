import pygimli as pg
import numpy as np
from settings import fpm

ntimes = 3

for timestep in range(1,ntimes+1):
    pd = pg.load("Output_file/mesh_file/paraDomain_1_t%s.bms" % timestep)
    resinv = np.loadtxt("Output_file/conventional_inv/res_conventional_t%s.dat" % timestep)
    vest = np.loadtxt("Output_file/conventional_inv/vel_conventional_t%s.dat" % timestep)

    fae, fie, fwe, maske = fpm.all(resinv, vest)
    print(np.min(fwe), np.max(fwe))
    np.savez("Output_file/conventional_inv/conventional_t%s.npz" % timestep, vel=np.array(vest), rho=np.array(resinv), fa=fae,
            fi=fie, fw=fwe, mask=maske)
