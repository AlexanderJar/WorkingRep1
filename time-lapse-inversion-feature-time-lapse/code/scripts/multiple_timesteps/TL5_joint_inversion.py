import sys

import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from fpinv import FourPhaseModel, JointInv, JointMod
from pygimli.physics import ERTManager
from pygimli.physics import TravelTimeManager
from pygimli.physics.traveltime.ratools import createGradientModel2D


# Iterate over time steps
ntimes = 3


# Settings
if len(sys.argv) > 1:
    scenario = "Fig2"
    poro = 0.3  # startmodel if poro is estimated
    fix_poro = False
    poro_min = 0.15
    poro_max = 0.45
    lam = 10
    delta = 100
    eps = 0
    case = 2
    zW = 1.0
else:
    scenario = "Fig1"
    fix_poro = True
    poro_min = 0
    poro_max = 1
    lam = 20
    case = 1
    zW = 1.0

############
# Settings
maxIter = 10
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

# Load inversion meshes
meshRST = pg.load("paraDomain_TL_%d.bms" % case)
meshERT = pg.load("meshERT_TL_%d.bms" % case)

if fix_poro:
    frtrue = np.load("true_model_TL.npz")["fr"]
    phi = []
    for cell in meshRST.cells():
        idx = mesh.findCell(cell.center()).id()
        phi.append(1 - frtrue[idx])
    phi = np.array(phi)
    fr_min = 0
    fr_max = 1
    # For multiple time steps
    phi_new = phi.copy()
    for t in range(ntimes):
        if t > 0:
            phi_new = np.concatenate((phi_new,phi))
    phi = phi_new.copy()
else:
    phi = poro

fpm = FourPhaseModel(phi=phi)


for t in range(1,ntimes+1):
    ttData = pg.DataContainer("tttrue_t%s.dat" % t, "s g")
    errors_tt = ttData("err") / ttData("t")
    tts = ttData("t")
    if t == 1:
        tts_new = tts.copy()
        errors_tt_new = errors_tt.copy()
    if t > 1:
        tts_new = pg.cat(tts_new,tts)
        errors_tt_new = pg.cat(errors_tt_new,errors_tt)
        #tts.append(ttData("t"))
        #errors_tt.append(ttData("err") / ttData("t"))

    
    ertScheme = pg.DataContainerERT("erttrue_t%s.dat" % t)
    rhoas = ertScheme("rhoa")
    errors_rhoa = ertScheme("err")
    if t == 1:
        rhoas_new = rhoas.copy()
        errors_rhoa_new = errors_rhoa.copy()
    if t > 1:
        rhoas_new = pg.cat(rhoas_new,rhoas)
        errors_rhoa_new = pg.cat(errors_rhoa_new,errors_rhoa)
        
    
# Setup managers and equip with meshes
ert = ERTManager()
ert.setMesh(meshERT)
ert.setData(ertScheme)

rst = TravelTimeManager(verbose=True)
rst.setData(ttData)
rst.setMesh(meshRST, secNodes=3)

# Setup joint modeling and inverse operators
JM = JointMod(meshRST, ert, rst, fpm, ntimes=ntimes, fix_poro=fix_poro, zWeight=zW)

data = pg.cat(tts_new,rhoas_new)
error = pg.cat(errors_tt_new, errors_rhoa_new)

# Set gradient starting model
veltrue = np.loadtxt("veltrue_TL.dat")
startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue[0:mesh.cellCount()]),
                                np.max(veltrue[0:mesh.cellCount()]))
                 

startmodel_new = startmodel.copy()
for t in range(ntimes):
    if t > 0:
        startmodel_new = np.concatenate((startmodel_new,startmodel))
startmodel = startmodel_new

velstart = 1 / startmodel

rhostart = np.ones_like(velstart[0:meshRST.cellCount()]) * np.mean(ertScheme("rhoa"))

rhostart_new = rhostart.copy()
for t in range(ntimes):
    if t > 0:
        rhostart_new = np.concatenate((rhostart_new,rhostart))
rhostart = rhostart_new

fas, fis, fws, _ = fpm.all(rhostart, velstart)
frs = np.ones_like(fas) - fpm.phi
if not fix_poro:
    frs[frs <= fr_min] = fr_min + 0.01
    frs[frs >= fr_max] = fr_max - 0.01
startmodel = np.concatenate((fws, fis, fas, frs))
# Fix small values to avoid problems in first iteration
startmodel[startmodel <= 0.01] = 0.01

inv = JointInv(JM, data, error, startmodel, lam=lam, delta=delta, eps=eps, frmin=fr_min,
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

np.savez("TL_joint_inversion_lam%s_zW%s_delta%s_eps%s_it%s_N%s.npz" % (lam,zW,delta,eps,maxIter,ntimes), vel=np.array(velest),
        rho=np.array(rhoest), fa=fae, fi=fie, fw=fwe, fr=fre, mask=array_mask)


print("#" * 80)
# ertchi, _ = JM.ERTchi2(model, error)
# rstchi, _ = JM.RSTchi2(model, error, tts)
ertchi, _ = JM.ERTchi2(model, error, rhoas_new)
rstchi, _ = JM.RSTchi2(model, error, tts_new)
print("ERT chi^2", ertchi)
print("RST chi^2", rstchi)
print("#" * 80)