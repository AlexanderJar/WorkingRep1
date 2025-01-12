import sys

import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ERTManager
from pygimli.physics import TravelTimeManager
from pygimli.physics.traveltime.ratools import createGradientModel2D


############
# Settings
maxIter = 4
zWeight = 1.
############

# Set case
case = int(sys.argv[1])

# Time-lapse scenario test case
# Iterate over time steps

ntimes = 3

for timestep in range(1, ntimes+1):

    ertData = pb.load("erttrue_t%s.dat" % timestep)
    print(ertData)
    mesh = pg.load("mesh_TL.bms")
    depth = mesh.ymax() - mesh.ymin()

    if timestep == 1:
        # Build inversion meshes
        # plc = mt.createParaMeshPLC(ertData, paraDepth=depth, paraDX=0.3, boundary=0,
        #                         paraBoundary=2)

        # coarse mesh
        plc = mt.createParaMeshPLC(ertData, paraDepth=depth, paraDX=1., boundary=0,
                                paraBoundary=2)

        if case == 1:
            # !!!adapt if you want to use case 1!!!
            for depth in (5, 15):
                start = plc.createNode(mesh.xmin(), -depth, 0.0)
                end = plc.createNode(mesh.xmax(), -depth, 0.0)
                plc.createEdge(start, end, marker=1)

        for sensor in ertData.sensorPositions():
            plc.createNode([sensor.x(), sensor.y() - 0.1])

        rect = mt.createRectangle([mesh.xmin(), mesh.ymin()],
                                [mesh.xmax(), mesh.ymax()], boundaryMarker=0)
        geom = mt.mergePLC([plc, rect])

        # meshRST = mt.createMesh(geom, quality=34, area=1, smooth=[1, 2])

        # coarse mesh
        meshRST = mt.createMesh(geom, quality=34, area=10, smooth=[1, 2])

        for cell in meshRST.cells():
            cell.setMarker(2)
        for boundary in meshRST.boundaries():
            boundary.setMarker(0)

        # Parameter Domain
        meshRST.save("paraDomain_TL_%s.bms" % case)

        # For ERT append boundary region
        meshERT = mt.appendTriangleBoundary(meshRST, xbound=500, ybound=500,
                                            quality=34, isSubSurface=True)
        meshERT.save("meshERT_TL_%s.bms" % case)

        # Load inversion meshes
        # meshRST = pg.load("paraDomain_TL_%d.bms" % case)
        # meshERT = pg.load("meshERT_TL_%d.bms" % case)

    # ERT inversion
    ert = ERTManager()
    ert.setMesh(meshERT)

    resinv = ert.invert(ertData, lam=30, zWeight=zWeight, maxIter=maxIter)
    print("ERT chi: %.2f" % ert.inv.chi2())
    print("ERT rms: %.2f" % ert.inv.inv.relrms())
    np.savetxt("res_conventional_TL_%d_t%s.dat" % (case, timestep), resinv)

    # Seismic inversion
    ttData = pg.DataContainer("tttrue_t%s.dat" % timestep, "s g")
    print(ttData)
    rst = TravelTimeManager(verbose=True)
    rst.setMesh(meshRST, secNodes=3)

    veltrue = np.loadtxt("veltrue_TL.dat")
    startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue[0:mesh.cellCount()]),
                                np.max(veltrue[0:mesh.cellCount()]))
    np.savetxt("rst_startmodel_TL%d_t%s.dat" % (case, timestep), 1 / startmodel)
    vest = rst.invert(ttData, zWeight=zWeight, startModel=startmodel,
                    maxIter=maxIter, lam=220)
    print("RST chi: %.2f" % rst.inv.chi2())
    print("RST rms: %.2f" % rst.inv.inv.relrms())

    rst.rayCoverage().save("rst_coverage_TL_%d_t%s.dat" % (case, timestep))
    np.savetxt("vel_conventional_TL_%d_t%s.dat" % (case, timestep), vest)
