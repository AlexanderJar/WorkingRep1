import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
import matplotlib.pyplot as plt
from fpinv import FourPhaseModel

# Model creation
ntimes = 3
nlayers = 4
ALT_thickness = [-5,-10,-15]

x_nodes = np.arange(0, 151, 1) # start, stop, step [m]
y_nodes = np.arange(-30, 1, 1)
mesh = mt.createGrid(x=x_nodes, y=y_nodes)

# Time-lapse scenario test case
# Iterate over time steps

rhotrue, veltrue, fa, fi, fw, fr = ([] for _ in range(6))

for timestep in range(1,ntimes+1):

    geom = mt.createWorld([0, -30], [150, 0], layers=ALT_thickness, worldMarker=False)
    geom.save("geom_TL_t%s.bms" % timestep)

    for cell in mesh.cells():
        if cell.center().y() > ALT_thickness[0]:
            cell.setMarker(0)
        if cell.center().y() > ALT_thickness[1] and cell.center().y() <= ALT_thickness[0]:
            if cell.center().x() <= 50:
                cell.setMarker(1)
            if cell.center().x() <= 100 and cell.center().x() > 50:
                cell.setMarker(2)
            elif cell.center().x() > 100:
                cell.setMarker(1)
        if cell.center().y() > ALT_thickness[2]  and cell.center().y() <= ALT_thickness[1]:
            if cell.center().x() <= 50:
                cell.setMarker(3)
            if cell.center().x() <= 100 and cell.center().x() > 50:
                cell.setMarker(4)
            elif cell.center().x() > 100:
                cell.setMarker(3)
        elif cell.center().y() <= ALT_thickness[2]:
            if cell.center().x() <= 50:
                cell.setMarker(5)
            if cell.center().x() <= 100 and cell.center().x() > 50:
                cell.setMarker(6)
            elif cell.center().x() > 100:
                cell.setMarker(5)
                
    for cell in mesh.cells():
        if cell.center().y() > ALT_thickness[0]:
            cell.setMarker(0)
        if cell.center().y() > ALT_thickness[1] and cell.center().y() <= ALT_thickness[0]:
            if cell.center().x() <= 75:
                cell.setMarker(1)
            elif cell.center().x() > 75:
                cell.setMarker(2)
        if cell.center().y() > ALT_thickness[2]  and cell.center().y() <= ALT_thickness[1]:
            if cell.center().x() <= 75:
                cell.setMarker(3)
            elif cell.center().x() > 75:
                cell.setMarker(4)
        elif cell.center().y() <= ALT_thickness[2]:
            if cell.center().x() <= 75:
                cell.setMarker(5)
            elif cell.center().x() > 75:
                cell.setMarker(6)
              

    pg.show(mesh, markers=True, showMesh=True)        
    plt.savefig('test_mesh_%s.png' % timestep)

    # Model creation based on pore fractions
    philayers = np.array([0.4, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2])
    frlayers = 1 - philayers
    
    if timestep == 1:
        fwlayers = np.array([0.3, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01])
        filayers = np.array([0.0, 0.2, 0.1, 0.28, 0.18, 0.28, 0.18])
    if timestep == 2:
        # fwlayers = np.array([0.3, 0.15, 0.1, 0.01, 0.01, 0.01, 0.01])
        fwlayers = np.array([0.3, 0.2, 0.1, 0.01, 0.01, 0.01, 0.01])
        filayers = np.array([0.0, 0.0, 0.0, 0.28, 0.18, 0.28, 0.18])
    if timestep == 3:
        # fwlayers = np.array([0.3, 0.15, 0.1, 0.15, 0.1, 0.01, 0.01])
        fwlayers = np.array([0.3, 0.2, 0.1, 0.2, 0.1, 0.01, 0.01])
        filayers = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.28, 0.18])

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