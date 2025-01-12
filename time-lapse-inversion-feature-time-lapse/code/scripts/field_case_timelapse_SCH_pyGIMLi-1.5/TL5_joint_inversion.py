import numpy as np

import pygimli as pg
from fpinv import FourPhaseModel, JointInv, JointMod
import pygimli.physics.ert as ertMan
# from pybert.manager import ERTManager
from pygimli.physics import TravelTimeManager, ERTManager
from pygimli.physics.traveltime import createGradientModel2D
from settings import *

import os     #added by Alex 25
import shutil     #added by Alex 25


try:     #added by Alex 25
    shutil.rmtree('Output_file/ji_inv')     #added by Alex 25
except:     #added by Alex 25
    print('Create JI inv file')     #added by Alex 25
os.mkdir('Output_file/ji_inv')     #added by Alex 25

timelapse = True
ntimes = 3

args = sys.argv
lam = 10.
eps = 0
delta = 100.
case = 1
zW = .25
gam = 0
maxIter = 10
atc = False
# weighting = False

if case == 2:
    case = 2
    constrained = True
    mesh = pg.load("Output_file/mesh_file/mesh_2.bms")
    paraDomain = pg.load("Output_file/mesh_file/paraDomain_2.bms")
else:
    case = 1
    constrained = False
    mesh = pg.load("Output_file/mesh_file/mesh_1_t1.bms")
    paraDomain = pg.load("Output_file/mesh_file/paraDomain_1_t1.bms")

pg.boxprint("Calculating case %s" % case)

# Load meshes and data
ertScheme = pg.DataContainerERT("Output_file/data_filtered/ert_filtered_t1.data")
ertScheme2 = pg.DataContainerERT("Output_file/data_filtered/ert_filtered_t2.data")
ertScheme3 = pg.DataContainerERT("Output_file/data_filtered/ert_filtered_t3.data")

fr_min = 0.1
fr_max = 0.9
phi = np.ones(paraDomain.cellCount()*ntimes) * poro
fpm = FourPhaseModel(phi=phi)

# Setup managers and equip with meshes
ert = ERTManager()
ert.setMesh(mesh)
ert.setData(ertScheme)
ert.fop.createRefinedForwardMesh()

ttData = pg.DataContainer("Output_file/data_filtered/rst_filtered_t1.data", "g s")
ttData2 = pg.DataContainer("Output_file/data_filtered/rst_filtered_t2.data", "g s")
ttData3 = pg.DataContainer("Output_file/data_filtered/rst_filtered_t3.data", "g s")

rst = TravelTimeManager(verbose=True)
rst.setData(ttData)
rst.setMesh(paraDomain)
rst.fop.createRefinedForwardMesh()

# Set errors
ttData.set("err", np.ones(ttData.size()) * rste)
ertScheme.set("err", np.ones(ertScheme.size()) * erte)
#ttData["k"] = ertMan.geometricFactors(ttData)
ertScheme["k"] = ertMan.geometricFactors(ertScheme)     #added by Alex 25

ttData2.set("err", np.ones(ttData2.size()) * rste)
ertScheme2.set("err", np.ones(ertScheme2.size()) * erte)
ertScheme2["k"] = ertMan.geometricFactors(ertScheme2)     #added by Alex 25

ttData3.set("err", np.ones(ttData3.size()) * rste)
ertScheme3.set("err", np.ones(ertScheme3.size()) * erte)
ertScheme3["k"] = ertMan.geometricFactors(ertScheme3)     #added by Alex 25

if constrained:
    # # Find cells around boreholes to fix ice content to zero
    fixcells = []
    # for cell in paraDomain.cells():
    #     x, y, _ = cell.center()
    #     if (x > 9) and (x < 11) and (y > -depth_5198):
    #         fixcells.append(cell.id())
    #     elif (x > 25) and (x < 27) and (y > -depth_5000):
    #         fixcells.append(cell.id())
    # fixcells = np.array(fixcells)
else:
    # Do not fix ice
    fixcells = False

# Setup joint modeling and inverse operators
JM = JointMod(paraDomain, ert, rst, fpm, ntimes=ntimes, fix_poro=False, zWeight=zW,
              fix_ice=fixcells)

tts = pg.cat(ttData("t"), ttData2("t"))
tts = pg.cat(tts, ttData3("t"))
rhoas = pg.cat(ertScheme("rhoa"), ertScheme2("rhoa"))
rhoas = pg.cat(rhoas, ertScheme3("rhoa"))
errors_tt = pg.cat(ttData("err") / ttData("t"), ttData2("err") / ttData2("t"))
errors_tt = pg.cat(errors_tt, ttData3("err") / ttData3("t"))
errors_rhoa = pg.cat(ertScheme("err"), ertScheme2("err"))
errors_rhoa = pg.cat(errors_rhoa, ertScheme3("err"))

# if weighting:
#     n_rst = ttData.size()*ntimes
#     n_ert = ertScheme.size()*ntimes
#     avg = (n_rst + n_ert) / 2
#     weight_rst = avg / n_rst
#     weight_ert = avg / n_ert
# else:
#     weight_rst = 1
#     weight_ert = 1

# errors_tt = pg.cat(ttData("err") / ttData("t") / weight_rst, ttData2("err") / ttData2("t") / weight_rst)
# errors_rhoa = pg.cat(ertScheme("err") / weight_ert, ertScheme2("err") / weight_ert)

data = pg.cat(tts,rhoas)
error = pg.cat(errors_tt, errors_rhoa)

minvel = 1000
maxvel = 5000
velstart = 1 / createGradientModel2D(ttData, paraDomain, minvel, maxvel)
velstart2 = 1 / createGradientModel2D(ttData2, paraDomain, minvel, maxvel)
velstart3 = 1 / createGradientModel2D(ttData3, paraDomain, minvel, maxvel)                               

velstart = np.concatenate((velstart, velstart2))
velstart = np.concatenate((velstart, velstart3))

rhostart = np.ones_like(velstart2) * np.mean(ertScheme("rhoa"))
rhostart2 = np.ones_like(velstart2) * np.mean(ertScheme2("rhoa"))
rhostart3 = np.ones_like(velstart2) * np.mean(ertScheme3("rhoa"))

rhostart = np.concatenate((rhostart, rhostart2))
rhostart = np.concatenate((rhostart, rhostart3))


fas, fis, fws, _ = fpm.all(rhostart, velstart)
frs = np.ones_like(fas) - fpm.phi
frs[frs <= fr_min] = fr_min + 0.01
frs[frs >= fr_max] = fr_max - 0.01
if fixcells is not False:
    fis[fixcells] = 0.0
startmodel = np.concatenate((fws, fis, fas, frs))

# Fix small values to avoid problems in first iteration
startmodel[startmodel <= 0.001] = 0.001

inv = JointInv(JM, data, error, startmodel, lam=lam, eps=eps, delta=delta, frmin=fr_min,
            frmax=fr_max, maxIter=maxIter, atc=atc)

# Run inversion
model = inv.run()
print(("Chi squared fit:", inv.getChi2()))

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

np.savez("Output_file/ji_inv/TL_joint_inversion_SCH_lam%s_zW%s_delta%s_gam%s_eps%s_it%s_new_gs.npz" % (lam,zW,delta,gam,eps,maxIter), vel=np.array(velest),
         rho=np.array(rhoest), fa=fae, fi=fie, fw=fwe, fr=fre, mask=array_mask)

fit = [inv.getChi2(),inv.relrms()]
np.savetxt("Output_file/ji_inv/fit_lam%s_zW%s_delta%s_eps%s_it%s_new_gs.txt" % (lam,zW,delta,eps,maxIter), fit)

print("#" * 80)
ertchi, _ = JM.ERTchi2(model, error, rhoas)
rstchi, _ = JM.RSTchi2(model, error, tts)
print("ERT chi^2", ertchi)
print("RST chi^2", rstchi)
print("#" * 80)
