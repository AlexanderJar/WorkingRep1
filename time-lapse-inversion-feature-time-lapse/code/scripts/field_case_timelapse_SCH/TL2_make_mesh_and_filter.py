import numpy as np

# import pybert as pb
import pygimli as pg
pg.verbose = print # temporary
import pygimli.meshtools as mt
from settings import erte, rste

ntimes = 3

ertData = []
rstData = []
maxrst = []

for timestep in range(1,ntimes+1):
    ertData.append(pg.DataContainer("ert_t%s.data" % timestep))
    print("Number of electrodes t%s:" % timestep, ertData[timestep-1].sensorCount())
    print(ertData[timestep-1])

    rstData.append(pg.DataContainer("rst_t%s.data" % timestep, "s g"))
    print("Number of shot/receivers t%s:" % timestep, rstData[timestep-1].sensorCount())
    maxrst.append(pg.max(pg.x(rstData[timestep-1].sensors())))

    idx = []
    for i, sensor in enumerate(ertData[timestep-1].sensors()):
        if sensor[0] >= 50.0:
            print(i)
            idx.append(i)

    #breakpoint()
    #ertData[timestep-1].removeSensorIdx(idx=idx)
    #ertData[timestep-1].removeInvalid()
    #ertData[timestep-1].removeUnusedSensors()
    ertData[timestep-1].set("err", pg.Vector(ertData[timestep-1].size(), erte))
    #ertData[timestep-1].save("ert_filtered_t%s.data" % timestep)

    rstData[timestep-1].set("err", pg.Vector(rstData[timestep-1].size(), rste))
    #
    # # Remove two data points with high v_a at zero-offset
    # Calculate offset
    px = pg.x(rstData[timestep-1].sensorPositions())
    gx = np.array([px[int(g)] for g in rstData[timestep-1]("g")])
    sx = np.array([px[int(s)] for s in rstData[timestep-1]("s")])
    offset = np.absolute(gx - sx)
    #va = offset / rstData[timestep-1]("t")
    rstData[timestep-1].markInvalid((offset < 5))
    #
    # # # Remove shot 27, too high apparent velocities
    # rstData.markInvalid(np.isclose(rstData("s"), 27))
    # rstData.markInvalid(210)  # outlier
    # rstData.removeInvalid()
    # rstData[timestep-1].markInvalid(rstData[timestep-1]("t") == 0.)
    rstData[timestep-1].removeInvalid()
    #rstData[timestep-1].save("rst_filtered_t%s.data" % timestep)
    # rstData = pg.DataContainer("rst_filtered.data", "s g")
    #########################
    # ertData = pb.load("ert_filtered_t%s.data" % timestep)
    print(ertData)

    print(len(ertData[timestep-1].sensorPositions()))
    for pos in ertData[timestep-1].sensorPositions():
        print(pos)

    def is_close(pos, data, tolerance=0.1):
        for posi in data.sensorPositions():
            dist = pos.dist(posi)
            if dist <= tolerance:
                return True
        return False


    combinedSensors = pg.DataContainer()
    for pos in ertData[timestep-1].sensorPositions():
        combinedSensors.createSensor(pos)

    for pos in rstData[timestep-1].sensorPositions():
        if is_close(pos, ertData[timestep-1]):
            print("Not adding", pos)
        else:
            combinedSensors.createSensor(pos)

    combinedSensors.sortSensorsX()
    x = pg.x(combinedSensors.sensorPositions()).array()
    z = pg.z(combinedSensors.sensorPositions()).array()

    np.savetxt("sensors_t%s.npy" % timestep, np.column_stack((x, z)))

    print("Number of combined positions t%s:" % timestep, combinedSensors.sensorCount())
    print(combinedSensors)
# %%

    combinedSensors = np.loadtxt("sensors_t1.npy")

    for case in 1, 2:
        # if case == 2:
        #     p1 = [8., 0, -0.03]
        #     p2 = [12., 0, -0.23]
        #     p3 = [24., 0, -0.37]
        #     p4 = [28., 0, -0.69]
        #     for p in p1, p2, p3, p4:
        #         combinedSensors.createSensor(p)
        #     combinedSensors.sortSensorsX()

        plc = mt.createParaMeshPLC(combinedSensors, paraDX=0.15, boundary=4,
                                paraDepth=20, paraBoundary=3,
                                paraMaxCellSize=0.3)

        # if case == 2:
        #     box = pg.Mesh(2)
        #     radius = 2.
        #     points = [(p1, p2), (p3, p4)]
        #     for x, depth, pts in zip([10., 26.], [depth_5198, depth_5000], points):
        #         start = plc.createNode(x - radius, -depth, 0.0)
        #         ul = plc.createNode(pts[0][0], pts[0][2], 0.0)
        #         b = plc.createEdge(start, ul, marker=20)
        #         box.copyBoundary(b)

        #         end = plc.createNode(x + radius, -depth, 0.0)
        #         ur = plc.createNode(pts[1][0], pts[1][2], 0.0)
        #         b = plc.createEdge(end, ur, marker=20)
        #         plc.createEdge(start, end, marker=1)
        #         box.copyBoundary(b)
        #         plc.addRegionMarker([x, -1], 2, 0.5)

        #     box.save("box.bms")

        # for x in [10., 26.]:
        #     plc.addRegionMarker([x, -12.0], 2, 0.5)

        mesh = mt.createMesh(plc, quality=33.8)

        # Set vertical boundaries of box to zero to allow lateral smoothing
        # if case == 2:
        #     for bound in mesh.boundaries():
        #         if bound.marker() == 20:
        #             bound.setMarker(0)

        mesh.save("mesh_%s_t%s.bms" % (case, timestep))

        # Extract inner domain where parameters should be estimated.
        # Outer domain is only needed for ERT forward simulation,
        # not for seismic traveltime calculations.
        paraDomain = pg.Mesh(2)
        paraDomain.createMeshByMarker(mesh, 2)
        paraDomain.save("paraDomain_%s_t%s.bms" % (case, timestep))
