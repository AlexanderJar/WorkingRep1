import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1

import matplotlib.pyplot as plt
#import pandas as pd

import pygimli as pg
import numpy as np


from fpinv import set_style
fs = 5.5
set_style(fs, style="seaborn-v0_8-dark")

mesh = pg.load("Output_file/mesh_file/paraDomain_1_t1.bms")
joint = np.load("Output_file/ji_inv/TL_joint_inversion_SCH_lam10.0_zW0.25_delta100.0_gam0_eps0_it10_new_gs.npz")
joint2 = np.load("Output_file/ji_inv/TL_joint_inversion_SCH_lam10.0_zW0.25_delta100.0_gam0_eps0_it10_new_gs.npz")

bl_08 = {
    "name": "SCH5198",
    "depth": 0.15,
    "pos": 10,
    "temp": [5.28275,3.83825,3.8665,3.623,2.79075,2.1915,1.66075,1.163,0.64125,0.1245,-0.1895,-0.45875,-0.64375,-0.715,-0.6455,-0.76525],
    "sensors": [0.2,0.4,0.8,1.2,1.6,2,2.5,3,3.5,4,5,7,9,10,11,13],
    "templim": (-1, 5),
    "alt": 4.5,
    "figname": "1D_profiles_2008.png",
    "timestep":1
}

bl_16 = {
    "name": "SCH5198",
    "depth": 0.15,
    "pos": 10,
    "temp": [5.59617,4.65942,3.65117,2.51233,1.61567,0.50775,-0.02333,-0.01833,0.12825,-0.03633,-0.01675,-0.08042,-0.13217,-0.14942,-0.17367],
    "sensors": [0.2,0.4,0.8,1.2,1.6,2,2.5,3,3.5,4,5,7,10,11,13],
    "templim": (-1, 5),
    "alt": 2.5,
    "figname": "1D_profiles_2016.png",
    "timestep":2
}

bl_17 = {
    "name": "SCH5198",
    "depth": 0.15,
    "pos": 10,
    "temp": [5.28275,3.83825,3.8665,3.623,2.79075,2.1915,1.66075,1.163,0.64125,0.1245,-0.1895,-0.45875,-0.64375,-0.715,-0.6455,-0.76525],
    "sensors": [0.2,0.4,0.8,1.2,1.6,2,2.5,3,3.5,4,5,7,9,10,11,13],
    "templim": (-1, 5),
    "alt": 4.0,
    "figname": "1D_profiles_2017.png",
    "timestep":3
}


def interpolate(mesh, data, x, y):
    result = []
    for yi in y[:-1]:
        #if yi > 1.7:
        #    cell = mesh.findCell(pg.Pos(x[0], yi))
        #    result.append(data[cell.id()])
        #else:
            cell = mesh.findCell(pg.Pos(x[0], yi))
            cell2 = mesh.findCell(pg.Pos(x[0]+0.1, yi+0.1))
            cell3 = mesh.findCell(pg.Pos(x[0]-0.1, yi+0.1))
            cell4 = mesh.findCell(pg.Pos(x[0]-0.1, yi-0.1))
            cell5 = mesh.findCell(pg.Pos(x[0]+0.1, yi-0.1))
            value = (data[cell.id()]+data[cell2.id()]+data[cell3.id()]+data[cell4.id()]+data[cell5.id()])/5
            result.append(value)
    return np.array(result)
        

def get(mesh, data, frac, bl):
    quant = data[frac]
    if "fr" in data:
        fr = data["fr"]
    else:
        fr = np.ones_like(data["fw"]) * (1 - 0.53)
    
    timestep = bl["timestep"]
    nCells = mesh.cellCount()
    quant = quant[nCells*(timestep-1):nCells*timestep]
    y = np.linspace(-15, -bl["depth"], 100)
    #profile = pg.interpolate(mesh, quant, x=np.ones_like(y) * bl["pos"], y=y)
    #fr_profile = pg.interpolate(mesh, fr, x=np.ones_like(y) * bl["pos"], y=y)
    profile = interpolate(mesh, quant, x=np.ones_like(y) * bl["pos"], y=y)
    fr_profile = interpolate(mesh, fr, x=np.ones_like(y) * bl["pos"], y=y)
    y += bl["depth"]
    if frac != "fr":
        profile /= (1-fr_profile)
    return (profile, -y[:-1])

short = ["T $\degree$C"]
short.extend([r"f$_{\rm %s}$ / (1 - f$_{\rm r}$)" % x for x in "wia"])
short.append(r"f$_{\rm r}$")
long = "     Temperature\n", "Water\nsaturation", "Ice\nsaturation", "Air\nsaturation", "Rock\ncontent"
colors = ["xkcd:red", "#0e59a2", "#684da1", "#127c39", "#bd3e02"]

bl = bl_17

fig, axs = plt.subplots(1, 5, sharey=True, figsize=(4.7, 1.7))

for ax, s, l, b in zip(axs, short, long, "abcde"):
    ax.axhline(bl["alt"], c="k", alpha=0.5)
   # ax.axhline(4.5, c="k", alpha=0.5)
   # ax.axhline(6.3, c="k", alpha=0.5)
    ax.set_xlabel(s)
    if b == "a":
        l += "     " + bl["name"]
    ax.set_title(l, fontsize=fs, weight="bold")  
    ax.set_title("(%s)\n" % b, loc="left", fontsize=fs, weight="bold")
     
temp = bl["temp"]
sens = bl["sensors"]

axs[0].axvline(0, c="k", ls="-", alpha=0.5)
axs[0].set_ylabel("Depth (m)")
axs[0].set_ylim(15, 0)
axs[0].set_xlim(*bl["templim"])
    
axs[0].plot(temp, sens, "-", c=colors[0])
    
#axs[0].plot(temp[:-1], sens[:-1], "-", c=colors[0])
#axs[0].plot(temp[-1], sens[-1], ".", c=colors[0], markersize=1.5)
    
for i, (ax, quant) in enumerate(zip(axs[1:], "wiar")):
        
    #if quant == "r":
    #    ax.axvline(1 - 0.4, c=colors[i + 1], ls=":")        
    #else:
    #    ax.plot(*get(mesh, conv, "f" + quant, bl), ":", c=colors[i + 1])
    
    ax.plot(*get(mesh, joint, "f" + quant, bl), ":", c=colors[i + 1])
    ax.plot(*get(mesh, joint2, "f" + quant, bl), "--", c=colors[i + 1])
        
    if quant in "ia":
        ax.axvline(0, c="k", ls="-", alpha=0.5)
    
    ax.plot([], [], "k:", label="TLPJI")
    ax.plot([], [], "k--", label="PJI")
    #plt.legend(bbox_to_anchor=(0.5, -.23), bbox_transform=plt.gcf().transFigure, ncol=3,
    #          frameon=False, loc="lower center")

fig.savefig('Output_file/'+bl["figname"]+'.png')
fig.show()