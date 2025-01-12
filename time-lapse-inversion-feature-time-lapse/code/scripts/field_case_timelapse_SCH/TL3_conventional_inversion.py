import numpy as np

# import pybert as pb
import pygimli as pg
pg.verbose = print # Temporary fix
import pygimli.meshtools as mt

from fpinv import FourPhaseModel, NN_interpolate
from pygimli.physics import ERTManager
from pygimli.physics import TravelTimeManager
from pygimli.physics.traveltime.ratools import createGradientModel2D
from pygimli.utils import boxprint
# from settings import *

############
# Settings
maxIter = 10
zWeight = .25
ntimes = 3
############

for timestep in range(1, ntimes+1):

    pg.boxprint("Timestep %s" % timestep)

    #need ertData, rstData, a mesh and phi to be given
    ertData = pg.load("ert_filtered_t%s.data" % timestep)
    print(ertData)
    mesh = pg.load("mesh_1_t%s.bms" % timestep)
    paraDomain = pg.load("paraDomain_1_t%s.bms" % timestep)
    depth = mesh.ymax() - mesh.ymin()
        
    ert = ERTManager()
    ert.setMesh(mesh)
    resinv = ert.invert(ertData, lam=60, zWeight=zWeight, maxIter=maxIter)
    print("ERT chi:", ert.inv.chi2())
    print("ERT rms:", ert.inv.relrms())

    np.savetxt("res_conventional_t%s.dat" % timestep, resinv)
    #############
    ttData = pg.DataContainer("rst_filtered_t%s.data" % timestep, "s g")
    rst = TravelTimeManager(verbose=True)

    # INVERSION
    rst.setMesh(mesh, secNodes=3)
    minvel = 1000
    maxvel = 5000
    startmodel = createGradientModel2D(ttData, paraDomain, minvel, maxvel)
    np.savetxt("rst_startmodel_t%s.dat" % timestep, 1 / startmodel)
    vest = rst.invert(ttData, mesh=paraDomain, zWeight=zWeight, lam=250)

    # vest = rst.inv.runChi1()
    print("RST chi:", rst.inv.chi2())
    print("RST rms:", rst.inv.relrms())

    rst.rayCoverage().save("rst_coverage_t%s.dat" % timestep)
    np.savetxt("vel_conventional_t%s.dat" % timestep, vest)
