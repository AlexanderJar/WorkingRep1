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
ntimes = 13
if timelapse == False:
    timestep = 3
    ntimes = 1

# Settings
poro = 0.3  # startmodel if poro is estimated
fix_poro = False
poro_min = 0.1
poro_max = 0.5
lam = 20.
eps = 0
delta = 0


############
# Settings
maxIter = 10
zW = 1.0
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

# Load inversion meshes
meshRST = pg.load("paraDomain_TL.bms")
meshERT = pg.load("meshERT_TL.bms")


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
        phi = np.tile(phi,ntimes)
else:
    phi = poro

fpm = FourPhaseModel(phi=phi)

# Setup managers and equip with meshes
ert = ERTManager()
ert.setMesh(meshERT)
ert.setData(ertScheme)

ttData = pg.DataContainer("tttrue_t1.dat", "s g")
rst = TravelTimeManager(verbose=True)
rst.setData(ttData)
rst.setMesh(meshRST, secNodes=3)

# Setup joint modeling and inverse operators
JM = JointMod(meshRST, ert, rst, fpm, ntimes=ntimes, fix_poro=fix_poro, zWeight=zW)

if timelapse == True:

    tts = ttData("t")
    rhoas = ertScheme("rhoa")
    errors_tt = ttData("err") / ttData("t")
    errors_rhoa = ertScheme("err")
    
    for timestep in range(2,ntimes+1): # start at t2, t1 already in ttData/ertScheme
        print(timestep)
        
        ttData2 = pg.DataContainer("tttrue_t%s.dat" % timestep, "s g")
        tts = pg.cat(tts, ttData2("t"))

        ertScheme2 = pg.DataContainerERT("erttrue_t%s.dat" % timestep)
        rhoas = pg.cat(rhoas, ertScheme2("rhoa"))

        errors_tt = pg.cat(errors_tt, ttData2("err") / ttData2("t"))

        errors_rhoa = pg.cat(errors_rhoa, ertScheme2("err"))

elif timelapse == False:

    ttData = pg.DataContainer("tttrue_t%s.dat" % timestep, "s g")
    ertScheme = pg.DataContainerERT("erttrue_t%s.dat" % timestep)

    tts = ttData("t")
    rhoas = ertScheme("rhoa")
    errors_tt = ttData("err") / ttData("t")
    errors_rhoa = ertScheme("err")

data = pg.cat(tts,rhoas)
error = pg.cat(errors_tt, errors_rhoa)

# Set gradient starting model
veltrue = np.loadtxt("veltrue_TL.dat")
startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue[0:mesh.cellCount()]),
                                 np.max(veltrue[0:mesh.cellCount()]))
# startmodel2 = createGradientModel2D(ttData2, meshRST, np.min(veltrue[mesh.cellCount():]),
#                                 np.max(veltrue[mesh.cellCount():]))
# startmodel3 = createGradientModel2D(ttData3, meshRST, np.min(veltrue[mesh.cellCount():]),
#                                 np.max(veltrue[mesh.cellCount():]))   

if timelapse == True:
    startmodel = np.tile(startmodel, ntimes)
#     startmodel = np.concatenate((startmodel, startmodel3))
# elif timelapse == False:
#     if timestep == 2:
#         startmodel = startmodel2
#     if timestep == 3:
#         startmodel = startmodel3

velstart = 1 / startmodel

rhostart = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme("rhoa"))

if timelapse == True:
#     rhostart2 = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme2("rhoa"))
#     rhostart = np.concatenate((rhostart, rhostart2))
#     rhostart3 = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme3("rhoa"))
#     rhostart = np.concatenate((rhostart, rhostart3))
    rhostart = np.tile(rhostart, ntimes)
# elif timelapse == False:
#     if timestep == 2:
#         rhostart = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme2("rhoa"))
#     if timestep == 3:
#         rhostart = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme3("rhoa"))

fas, fis, fws, _ = fpm.all(rhostart, velstart)
frs = np.ones_like(fas) - fpm.phi
if not fix_poro:
    frs[frs <= fr_min] = fr_min + 0.01
    frs[frs >= fr_max] = fr_max - 0.01
startmodel = np.concatenate((fws, fis, fas, frs))
# Fix small values to avoid problems in first iteration
startmodel[startmodel <= 0.01] = 0.01

# breakpoint()

inv = JointInv(JM, data, error, startmodel, lam=lam, eps=eps, delta=delta, frmin=fr_min,
            frmax=fr_max, maxIter=maxIter)
inv.setModel(startmodel)

# Run inversion
model = inv.run()
pg.boxprint(("Chi squared fit:", inv.getChi2()), sym="+")

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
    np.savez("TL_joint_inversion_lam%s_zW%s_delta%s_eps%s_it%s.npz" % (lam,zW,delta,eps,maxIter), vel=np.array(velest),
            rho=np.array(rhoest), fa=fae, fi=fie, fw=fwe, fr=fre, mask=array_mask)
if timelapse == False:
    np.savez("TL_joint_inversion_lam%s_zW%s_delta%s_eps%s_it%s_t%s.npz" % (lam,zW,delta,eps,maxIter,timestep), vel=np.array(velest),
            rho=np.array(rhoest), fa=fae, fi=fie, fw=fwe, fr=fre, mask=array_mask)

fit = [inv.getChi2(),inv.relrms()]
np.savetxt("fit_lam%s_zW%s_delta%s_eps%s_it%s.txt" % (lam,zW,delta,eps,maxIter), fit)

print("#" * 80)
# ertchi, _ = JM.ERTchi2(model, error)
# rstchi, _ = JM.RSTchi2(model, error, tts)
ertchi, _ = JM.ERTchi2(model, error, rhoas)
rstchi, _ = JM.RSTchi2(model, error, tts)
print("ERT chi^2", ertchi)
print("RST chi^2", rstchi)
print("#" * 80)