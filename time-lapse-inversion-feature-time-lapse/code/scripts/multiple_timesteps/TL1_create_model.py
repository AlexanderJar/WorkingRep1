import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
import matplotlib.pyplot as plt
from fpinv import FourPhaseModel

import random

# Model creation
ntimes = 3
ALT_thickness = [-15]

x_nodes = np.arange(0, 151, 1) # start, stop, step [m]
y_nodes = np.arange(-30, 1, 1)
mesh = mt.createGrid(x=x_nodes, y=y_nodes)

# Possible combinations wiar
comb1 = [0.02, 0.36, 0.02, 0.6]
comb2 = [0.02, 0.16, 0.02, 0.8]

comb3 = [0.38, 0.0, 0.02, 0.6]
comb4 = [0.18, 0.0, 0.02, 0.8]

comb5 = [0.1, 0.0, 0.3, 0.6]
comb6 = [0.05, 0.0, 0.15, 0.8]

comb7 = [0.2, 0.18, 0.02, 0.6]
comb8 = [0.1, 0.08, 0.02, 0.8]

comb9 = [0.2, 0.0, 0.2, 0.6]
comb10 = [0.1, 0.0, 0.1, 0.8]

combs_r06 = [comb1,comb3,comb5,comb7,comb9]
combs_r08 = [comb2,comb4,comb6,comb8,comb10]


# Iterate over time steps

rhotrue, veltrue, fa, fi, fw, fr = ([] for _ in range(6))

for timestep in range(1,ntimes+1):

    # geom = mt.createWorld([0, -30], [150, 0], layers=ALT_thickness, worldMarker=False)
    geom = mt.createRectangle(start=[0,0], end=[75,-15]) + mt.createRectangle(start=[75,0], end=[150,-15])
    geom += mt.createRectangle(start=[0,-15], end=[75,-30]) + mt.createRectangle(start=[75,-15], end=[150,-30])
    geom.save("geom_TL_t%s.bms" % timestep)

    for cell in mesh.cells():
        if cell.center().y() > ALT_thickness[0]:
            if cell.center().x() <= 75:
                cell.setMarker(0)
            elif cell.center().x() > 75:
                cell.setMarker(1)
        if cell.center().y() <= ALT_thickness[0]:
            if cell.center().x() <= 75:
                cell.setMarker(2)
            elif cell.center().x() > 75:
                cell.setMarker(3)
        
    pg.show(mesh, markers=True, showMesh=True)        
    # plt.savefig('test_mesh_%s.png' % timestep)

    # Model creation based on pore fractions
    philayers = np.array([0.4, 0.2, 0.4, 0.2])
    frlayers = 1 - philayers

    b1 = random.choice(combs_r06)
    b2 = random.choice(combs_r08)
    b3 = random.choice(combs_r06)
    b4 = random.choice(combs_r08)

    fwlayers = np.array([b1[0], b2[0], b3[0], b4[0]])
    filayers = np.array([b1[1], b2[1], b3[1], b4[1]])

    falayers = philayers - fwlayers - filayers

    falayers[np.isclose(falayers, 0.0)] = 0.0
    print(falayers)

    # Save for covariance calculations
    Fsyn = np.vstack((fwlayers, filayers, falayers, frlayers))
    np.savetxt("syn_model_t%s.dat" % timestep, Fsyn)

    fpm = FourPhaseModel(phi=philayers)

    print(falayers + filayers + fwlayers + frlayers)
    rholayers = fpm.rho(fwlayers, filayers, falayers, frlayers)
    vellayers = 1. / fpm.slowness(fwlayers, filayers, falayers, frlayers)

    print(rholayers)
    print(vellayers)

    def to_mesh(data):
        return data[mesh.cellMarkers()]

    # Append values for every time step
    rhotrue = np.append(rhotrue, to_mesh(rholayers))
    veltrue = np.append(veltrue, to_mesh(vellayers))

    # %%
    # Save sensors, true model and mesh
    # Append values for every time step
    fa = np.append(fa, to_mesh(falayers))
    fi = np.append(fi, to_mesh(filayers))
    fw = np.append(fw, to_mesh(fwlayers))
    fr = np.append(fr, to_mesh(frlayers))

    fpm.fr = fr
    fpm.phi = 1 - fr
    # fpm.show(mesh, rhotrue, veltrue)

    assert np.allclose(fa + fi + fw + fr, 1)

# Save files that contain data for all time steps at once
np.savez("true_model_TL.npz", rho=rhotrue, vel=veltrue, fa=fa, fi=fi, fw=fw,
    fr=fr)

sensors = np.arange(10, 141, 2.5, dtype="float")
print(sensors)
print("Sensors", len(sensors))
sensors.dump("sensors_TL.npy")

mesh.save("mesh_TL.bms")
np.savetxt("rhotrue_TL.dat", rhotrue)
np.savetxt("veltrue_TL.dat", veltrue)