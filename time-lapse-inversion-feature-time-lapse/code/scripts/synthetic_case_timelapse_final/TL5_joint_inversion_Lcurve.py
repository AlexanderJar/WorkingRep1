import sys

import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from fpinv import FourPhaseModel, JointInv, JointMod
from pygimli.physics import ERTManager
from pygimli.physics import TravelTimeManager
from pygimli.physics.traveltime.ratools import createGradientModel2D


# Time-lapse scenario test case
# Iterate over time steps
timelapse = True
ntimes = 3
if timelapse == False:
    timestep = 1
    ntimes = 1

# Settings
poro = 0.3  # startmodel if poro is estimated
fix_poro = False
poro_min = 0.1
poro_max = 0.5
lams = [0.1,1,10,20,100]
deltas = [0,1,10,100,150,200,1000]
eps=0
zW = 1.0

############
# Settings
maxIter = 4
############

# Poro to rock content (inversion parameter)
fr_min = 1 - poro_max
fr_max = 1 - poro_min

# Load meshes and data
mesh = pg.load("mesh_TL.bms")
depth = mesh.ymax() - mesh.ymin()
true = np.load("true_model_TL.npz")
sensors = np.load("sensors_TL.npy", allow_pickle=True)

veltrue, rhotrue, fatrue, fitrue, fwtrue = true["vel"], true["rho"], true[
    "fa"], true["fi"], true["fw"]

ertScheme = pg.DataContainerERT("erttrue_t1.dat")
ertScheme2 = pg.DataContainerERT("erttrue_t2.dat")
ertScheme3 = pg.DataContainerERT("erttrue_t3.dat")

# Load inversion meshes
meshRST = pg.load("paraDomain_TL.bms")
meshERT = pg.load("meshERT_TL.bms")

# # Build inversion meshes
# # plc = mt.createParaMeshPLC(ertScheme, paraDepth=depth, paraDX=0.3, boundary=0,
# #                         paraBoundary=2)

# # coarse mesh
# plc = mt.createParaMeshPLC(ertScheme, paraDepth=depth, paraDX=1., boundary=0,
#                         paraBoundary=2)

# for sensor in ertScheme.sensorPositions():
#     plc.createNode([sensor.x(), sensor.y() - 0.1])

# rect = mt.createRectangle([mesh.xmin(), mesh.ymin()],
#                         [mesh.xmax(), mesh.ymax()], boundaryMarker=0)
# geom = mt.mergePLC([plc, rect])

# # meshRST = mt.createMesh(geom, quality=34, area=1, smooth=[1, 2])

# # coarse mesh
# meshRST = mt.createMesh(geom, quality=34, area=10, smooth=[1, 2])

# for cell in meshRST.cells():
#     cell.setMarker(2)
# for boundary in meshRST.boundaries():
#     boundary.setMarker(0)

# meshRST.save("paraDomain_TL.bms")

# meshERT = mt.appendTriangleBoundary(meshRST, xbound=500, ybound=500,
#                                     quality=34, isSubSurface=True)
# meshERT.save("meshERT_TL.bms")

if fix_poro:
    frtrue = np.load("true_model_TL.npz")["fr"]
    phi = []
    for cell in meshRST.cells():
        idx = mesh.findCell(cell.center()).id()
        phi.append(1 - frtrue[idx])
    phi = np.array(phi)
    fr_min = 0
    fr_max = 1
    # For 3 time steps
    if timelapse == True:
        phi = np.concatenate((phi,phi,phi))
else:
    phi = poro

fpm = FourPhaseModel(phi=phi)

# Setup managers and equip with meshes
ert = ERTManager()
ert.setMesh(meshERT)
ert.setData(ertScheme)

ttData = pg.DataContainer("tttrue_t1.dat", "s g")
ttData2 = pg.DataContainer("tttrue_t2.dat", "s g")
ttData3 = pg.DataContainer("tttrue_t3.dat", "s g")
rst = TravelTimeManager(verbose=True)
rst.setData(ttData)
rst.setMesh(meshRST, secNodes=3)

# Setup joint modeling and inverse operators
JM = JointMod(meshRST, ert, rst, fpm, ntimes=ntimes, fix_poro=fix_poro, zWeight=zW)

if timelapse == True:
    tts = pg.cat(ttData("t"), ttData2("t"))
    tts = pg.cat(tts, ttData3("t"))
    rhoas = pg.cat(ertScheme("rhoa"), ertScheme2("rhoa"))
    rhoas = pg.cat(rhoas, ertScheme3("rhoa"))
    errors_tt = pg.cat(ttData("err") / ttData("t"), ttData2("err") / ttData2("t"))
    errors_tt = pg.cat(errors_tt, ttData3("err") / ttData3("t"))
    errors_rhoa = pg.cat(ertScheme("err"), ertScheme2("err"))
    errors_rhoa = pg.cat(errors_rhoa, ertScheme3("err"))
elif timelapse == False:
    if timestep == 1:
        tts = ttData("t")
        rhoas = ertScheme("rhoa")
        errors_tt = ttData("err") / ttData("t")
        errors_rhoa = ertScheme("err")
    elif timestep == 2:
        tts = ttData2("t")
        rhoas = ertScheme2("rhoa")
        errors_tt = ttData2("err") / ttData2("t")
        errors_rhoa = ertScheme2("err")
    elif timestep == 3:
        tts = ttData3("t")
        rhoas = ertScheme3("rhoa")
        errors_tt = ttData3("err") / ttData3("t")
        errors_rhoa = ertScheme3("err")
data = pg.cat(tts,rhoas)
error = pg.cat(errors_tt, errors_rhoa)

# Set gradient starting model
veltrue = np.loadtxt("veltrue_TL.dat")
startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue[0:mesh.cellCount()]),
                                np.max(veltrue[0:mesh.cellCount()]))
startmodel2 = createGradientModel2D(ttData2, meshRST, np.min(veltrue[mesh.cellCount():]),
                                np.max(veltrue[mesh.cellCount():]))
startmodel3 = createGradientModel2D(ttData3, meshRST, np.min(veltrue[mesh.cellCount():]),
                                np.max(veltrue[mesh.cellCount():]))                               
if timelapse == True:
    startmodel = np.concatenate((startmodel, startmodel2))
    startmodel = np.concatenate((startmodel, startmodel3))
elif timelapse == False:
    if timestep == 2:
        startmodel = startmodel2
    if timestep == 3:
        startmodel = startmodel3

velstart = 1 / startmodel

rhostart = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme("rhoa"))

if timelapse == True:
    rhostart2 = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme2("rhoa"))
    rhostart = np.concatenate((rhostart, rhostart2))
    rhostart3 = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme3("rhoa"))
    rhostart = np.concatenate((rhostart, rhostart3))
elif timelapse == False:
    if timestep == 2:
        rhostart = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme2("rhoa"))
    if timestep == 3:
        rhostart = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme3("rhoa"))

fas, fis, fws, _ = fpm.all(rhostart, velstart)
frs = np.ones_like(fas) - fpm.phi
if not fix_poro:
    frs[frs <= fr_min] = fr_min + 0.01
    frs[frs >= fr_max] = fr_max - 0.01
startmodel = np.concatenate((fws, fis, fas, frs))
# Fix small values to avoid problems in first iteration
startmodel[startmodel <= 0.01] = 0.01

# 2D L-Curve: try different lambdas and deltas

chis = np.zeros((len(lams),len(deltas)))
rmss = np.zeros((len(lams),len(deltas)))

i,j = 0,0
for lam in lams:
    j = 0
    for delta in deltas:

        inv = JointInv(JM, data, error, startmodel, lam=lam, delta=delta, eps=eps, frmin=fr_min,
                    frmax=fr_max, maxIter=maxIter)
        inv.setModel(startmodel)

        # Run inversion
        model = inv.run()
        pg.boxprint(("Chi squared fit:", inv.getChi2()), sym="+")

        chis[i,j] = inv.getChi2()
        rmss[i,j] = inv.relrms()

        # Save results
        fwe, fie, fae, fre = JM.fractions(model)
        fsum = fwe + fie + fae + fre

        print("Min/Max sum:", min(fsum), max(fsum))

        rhoest = JM.fpm.rho(fwe, fie, fae, fre)
        velest = 1. / JM.fpm.slowness(fwe, fie, fae, fre)

        array_mask = np.array(((fae < 0) | (fae > 1 - fre))
                            | ((fie < 0) | (fie > 1 - fre))
                            | ((fwe < 0) | (fwe > 1 - fre))
                            | ((fre < 0) | (fre > 1))
                            | (fsum > 1.01))

        if timelapse == True:
            np.savez("TL_joint_inversion_Lcurve_lam%s_zW%s_delta%s_eps%s_it%s.npz" % (lam,zW,delta,eps,maxIter), vel=np.array(velest),
                    rho=np.array(rhoest), fa=fae, fi=fie, fw=fwe, fr=fre, mask=array_mask)
        if timelapse == False:
            np.savez("TL_joint_inversion_t%s.npz" % timestep, vel=np.array(velest),
                    rho=np.array(rhoest), fa=fae, fi=fie, fw=fwe, fr=fre, mask=array_mask)

        print("#" * 80)
        # ertchi, _ = JM.ERTchi2(model, error)
        # rstchi, _ = JM.RSTchi2(model, error, tts)
        ertchi, _ = JM.ERTchi2(model, error, rhoas)
        rstchi, _ = JM.RSTchi2(model, error, tts)
        print("ERT chi^2", ertchi)
        print("RST chi^2", rstchi)
        print("#" * 80)

        j+=1
        print(chis,rmss)
    i+=1

np.save("chi.npy",chis,allow_pickle=True)
np.save("rms.npy",rmss,allow_pickle=True)
np.save("deltas.npy",deltas,allow_pickle=True)
np.save("lambdas.npy",lams,allow_pickle=True)