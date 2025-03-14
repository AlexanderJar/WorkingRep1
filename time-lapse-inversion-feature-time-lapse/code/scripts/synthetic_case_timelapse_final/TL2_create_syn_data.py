import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt

from pygimli.physics import ERTManager
from pygimli.physics import TravelTimeManager
from pygimli.physics.traveltime import createRAData


# Time-lapse scenario test case
# Iterate over time steps

mesh = pg.load("mesh_TL.bms")
sensors = np.load("sensors_TL.npy", allow_pickle=True)
rhotrue = np.loadtxt("rhotrue_TL.dat")
veltrue = np.loadtxt("veltrue_TL.dat")

# Create more realistic data set
ertScheme = pb.createData(sensors, "dd", spacings=[1,2,4])
k = pb.geometricFactors(ertScheme)
ertScheme.markInvalid(pg.abs(k) > 5000)
ertScheme.removeInvalid()

ert = ERTManager()

# Create suitable mesh for ert forward calculation
# NOTE: In the published results paraMaxCellSize=1.0 was used, which is
# increased here to allow testing on Continuous Integration services.
meshERTFWD = mt.createParaMesh(ertScheme, quality=33.5, paraMaxCellSize=1.0,
                            paraDX=0.2, boundaryMaxCellSize=50,
                            paraBoundary=30)
pg.show(meshERTFWD)

# Create mesh for rst forward calculation
meshRSTFWD = pg.Mesh()
meshRSTFWD.createMeshByMarker(meshERTFWD, 2)

# Check number of time steps
ntimes = int(len(rhotrue)/mesh.cellCount())

for timestep in range(1,ntimes+1):

    pg.boxprint("Simulate apparent resistivities for time step %s" % timestep)
    res = pg.Vector()
    pg.interpolate(mesh, rhotrue[(timestep-1)*mesh.cellCount():timestep*mesh.cellCount()], meshERTFWD.cellCenters(), res)
    res = mt.fillEmptyToCellArray(meshERTFWD, res, slope=True)
    ert.setMesh(meshERTFWD)
    ert.fop.createRefinedForwardMesh()
    ertData = ert.simulate(meshERTFWD, res, ertScheme, noiseLevel=0.05,
                        noiseAbs=0.0)
    ertData.save("erttrue_t%s.dat" % timestep)
    ert.setData(ertData)
    ert.setMesh(meshERTFWD)
    ert.inv.dataVals = ertData("rhoa")

    pg.boxprint("Simulate traveltimes for time step %s" % timestep)
    vel = pg.Vector()
    pg.interpolate(mesh, veltrue[(timestep-1)*mesh.cellCount():timestep*mesh.cellCount()], meshRSTFWD.cellCenters(), vel)
    vel = mt.fillEmptyToCellArray(meshRSTFWD, vel, slope=False)

    ttScheme = createRAData(sensors)
    rst = TravelTimeManager(verbose=True)

    error = 0.0005 # = 0.5 ms
    meshRSTFWD.createSecondaryNodes(3)
    ttData = rst.simulate(mesh=meshRSTFWD, slowness=1. / vel, scheme=ttScheme,
                        noisify=True, noiseLevel=0.0, noiseAbs=error, seed=1234)
    ttData.set("err", np.ones(ttData.size()) * error)

    # remove negative traveltimes
    tts = ttData.get("t")
    new_tts = []
    for tt in tts:
        if tt <= 0.0:
            tt = 0.0005
            print("Removed negative traveltime.")
        new_tts.append(tt)

    ttData.set("t", new_tts)

    rst.setData(ttData)
    rst.fop.data.save("tttrue_t%s.dat" % timestep)
