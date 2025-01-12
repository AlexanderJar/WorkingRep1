import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from fpinv import FourPhaseModel

# Model creation
world = mt.createWorld([0, -30], [150, 0], layers=[-5, -15], worldMarker=False)
block = mt.createPolygon([[60, -5], [90, -5], [100, -15], [50, -15]],
                         isClosed=True)
geom = mt.mergePLC([world, block])
geom.addRegionMarker((80, -2.5), 0)
geom.addRegionMarker((20, -10), 1)
geom.addRegionMarker((80, -10), 2)
geom.addRegionMarker((120, -10), 3)
geom.addRegionMarker((80, -25), 4)
geom.save("geom_TL.bms")

mesh = mt.createMesh(geom, area=1.0)

# pg.show(mesh, markers=True)

# Time-lapse scenario test case
# Iterate over time steps

ntimes = 3
rhotrue, veltrue, fa, fi, fw, fr = ([] for _ in range(6))

for timestep in range(1,ntimes+1):

    # Model creation based on pore fractions
    philayers = np.array([0.4, 0.3, 0.3, 0.3, 0.2])
    frlayers = 1 - philayers

    if timestep == 1:
        fwlayers = np.array([0.3, 0.18, 0.02, 0.1, 0.02])
        filayers = np.array([0.0, 0.1, 0.28, 0.18, 0.18])
    if timestep == 2:
        fwlayers = np.array([0.3, 0.18, 0.06, 0.1, 0.02])
        filayers = np.array([0.0, 0.1, 0.23, 0.18, 0.18])
    if timestep == 3:
        fwlayers = np.array([0.3, 0.18, 0.1, 0.1, 0.02])
        filayers = np.array([0.0, 0.1, 0.18, 0.18, 0.18])

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
