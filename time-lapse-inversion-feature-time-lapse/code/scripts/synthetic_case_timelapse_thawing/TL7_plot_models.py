import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import ImageGrid

import pygimli as pg
from fpinv import add_inner_title, logFormat, rst_cov, set_style
from pygimli.mplviewer import drawModel


fs = 5.5
set_style(fs, style="seaborn-dark")

# Time-lapse scenario test case
# Iterate over time steps

ntimes = 3
case = 2

mesh = pg.load("mesh_TL.bms")
sensors = np.load("sensors_TL.npy", allow_pickle=True)

nCells = mesh.cellCount()


veltrue_t = np.zeros([nCells, ntimes])
rhotrue_t = np.zeros([nCells, ntimes])
fa_t = np.zeros([nCells, ntimes])
fi_t = np.zeros([nCells, ntimes])
fw_t = np.zeros([nCells, ntimes])
fr_t = np.zeros([nCells, ntimes])

sumtrue_t = np.zeros([nCells, ntimes])
ritrue_t = np.zeros([nCells, ntimes])

    
for timestep in range(1,ntimes+1):

    # Load data
    true = np.load("true_model_TL.npz")

    # True model
    veltrue, rhotrue, fa, fi, fw, fr = true["vel"], true["rho"], true["fa"], true[
        "fi"], true["fw"], true["fr"]

    veltrue_t[:,timestep-1] = veltrue[nCells*(timestep-1):nCells*timestep]
    rhotrue_t[:,timestep-1] = rhotrue[nCells*(timestep-1):nCells*timestep]
    fa_t[:,timestep-1] = fa[nCells*(timestep-1):nCells*timestep]
    fi_t[:,timestep-1] = fi[nCells*(timestep-1):nCells*timestep]
    fw_t[:,timestep-1] = fw[nCells*(timestep-1):nCells*timestep]
    fr_t[:,timestep-1] = fr[nCells*(timestep-1):nCells*timestep]

    sumtrue_t[:,timestep-1] = fa_t[:,timestep-1] + fi_t[:,timestep-1] + fw_t[:,timestep-1] + fr_t[:,timestep-1]
    
    # Rock and ice
    ritrue_t[:,timestep-1] = fi_t[:,timestep-1] + fr_t[:,timestep-1]

    # Some helper functions
    def update_ticks(cb, log=False, label="", cMin=None, cMax=None):
        t = ticker.FixedLocator([cMin, cMax])
        cb.set_ticks(t)
        ticklabels = cb.ax.yaxis.get_ticklabels()
        for i, tick in enumerate(ticklabels):
            if i == 0:
                tick.set_verticalalignment("bottom")
            if i == len(ticklabels) - 1:
                tick.set_verticalalignment("top")

        cb.ax.annotate(label, xy=(1, 0.5), xycoords='axes fraction',
                    xytext=(10, 0), textcoords='offset pixels',
                    horizontalalignment='center', verticalalignment='center',
                    rotation=90, fontsize=fs, fontweight="regular")


    def lim(data):
        """Return appropriate colorbar limits."""
        data = np.array(data)
        print("dMin", data.min(), "dMax", data.max())
        if data.min() < 0:
            dmin = 0.0
        else:
            dmin = np.around(data.min(), 2)
        dmax = np.around(data.max(), 2)
        kwargs = {"cMin": dmin, "cMax": dmax}
        return kwargs


    def draw(ax, mesh, model, **kwargs):
        model = np.array(model)
        tol = 1e-3  # do not hide values that are very close (but below) zero
        if not np.isclose(model.min(), 0.0, atol=tol) and (model < 0).any():
            model = np.ma.masked_where(model < -tol, model)

        if "coverage" in kwargs:
            model = np.ma.masked_where(kwargs["coverage"] == 0, model)
        gci = drawModel(ax, mesh, model, rasterized=True, nLevs=2, **kwargs)
        return gci


    def minmax(data):
        """Return minimum and maximum of data as a 2-line string."""
        tmp = np.array(data)
        print("max", tmp.max())
        min = tmp.min()
        if np.max(tmp) > 10 and np.max(tmp) < 1e4:
            return "min: %d | max: %d" % (min, tmp.max())
        if np.max(tmp) > 1e4:
            return "min: %d" % min + " | max: " + logFormat(tmp.max())
        else:
            if min < 0:
                min = ("%.3f" % min).rstrip("0")
            else:
                min = ("%.2f" % min).rstrip("0")
            min = min.rstrip(".")
            max = ("%.2f" % tmp.max()).rstrip("0")
            return "min: %s | max: %s" % (min, max)


# %%
fig = plt.figure(figsize=(10, 6))
grid = ImageGrid(fig, 111, nrows_ncols=(8, 3), axes_pad=[0.03, 0.03],
                share_all=True, add_all=True, cbar_location="right",
                cbar_mode="edge", cbar_size="5%", cbar_pad=0.05, aspect=True)

labels = ["v (m/s)", r" $\rho$ ($\Omega$m)"]
labels.extend([r"f$_{\rm %s}$" % x for x in "wiar"])
labels.extend([r"$\Sigma$ f$_{\rm j}$", r"f$_{\rm i}$ + f$_{\rm r}$"])

long_labels = [
    "Velocity", "Res.", "Water content", "Ice content", "Air content",
    "Rock content", "Sum all", "Sum ice and rock"
]
meshs = [mesh, mesh, mesh]
cmaps = ["viridis", "Spectral_r", "Blues", "Purples", "Greens", "Oranges", "RdBu","RdPu"]
datas = [(veltrue_t[:,0], veltrue_t[:,1], veltrue_t[:,2]), (rhotrue_t[:,0], rhotrue_t[:,1], rhotrue_t[:,2]),
        (fw_t[:,0], fw_t[:,1], fw_t[:,2]), (fi_t[:,0], fi_t[:,1], fi_t[:,2]), (fa_t[:,0], fa_t[:,1], fa_t[:,2]), 
        (fr_t[:,0], fr_t[:,1], fr_t[:,2]), (sumtrue_t[:,0], sumtrue_t[:,1], sumtrue_t[:,2]),(ritrue_t[:,0], ritrue_t[:,1], ritrue_t[:,2])]

for i, (row, data, label,
        cmap) in enumerate(zip(grid.axes_row, datas, labels, cmaps)):
    print("Plotting", label)
    borderpad = 0.2
    if i == 0:
        lims = {"cMin": 1500, "cMax": 5000}
    elif i == 1:
        lims = {"cMin": 1e3, "cMax": 200000}
        borderpad = 0.07
    elif i == 2:
        lims = {"cMin": 0.00000001, "cMax": 0.35}
    elif i == 3:
        lims = {"cMin": 0.00000001, "cMax": 0.35}
    elif i == 4:
        lims = {"cMin": 0.00000001, "cMax": 0.3}
    elif i == 5:
        lims = {"cMin": 0.6, "cMax": 0.8}
    elif i == 6:
        lims = {"cMin": 0.95, "cMax": 1.05}
    elif i == 7:
        lims = {"cMin": 0.6000001, "cMax": 1.0}
    else:
        lims = lim(
            list(data[0]) + list(data[1]) + list(data[2]))
    print(lims)
    logScale = True if "rho" in label else False
    ims = []
    for j, ax in enumerate(row):
        coverage = np.ones(mesh.cellCount())
        color = "k" if i not in (1, 3, 5, 7) else "w"
        ims.append(
            draw(ax, meshs[j], data[j], **lims, logScale=logScale,
                coverage=coverage))
        # ax.text(0.987, 0.05, minmax(data[j]), transform=ax.transAxes, fontsize=fs,
        #         ha="right", color=color)
        add_inner_title(ax, minmax(data[j][coverage > 0]), loc=4, size=fs,
                        fw="regular", frame=False, c=color,
                        borderpad=borderpad)
        ims[j].set_cmap(cmap)

    cb = fig.colorbar(ims[0], cax=grid.cbar_axes[i])
    update_ticks(cb, log=logScale, label=label, **lims)

for ax, title, num in zip(grid.axes_row[0], [
        "$T_1$", "$T_2$",
        "$T_3$"
], "abc"):
    ax.set_title(title, fontsize=fs + 1, fontweight="bold")
    ax.set_title("(%s)" % num, loc="left", fontsize=fs + 1, fontweight="bold")


def add_labs_to_col(col, labs):
    for ax, lab in zip(grid.axes_column[col], labs):
        if lab != "Inverted":
            weight = "regular"
            c = "w"
            add_inner_title(ax, lab, loc=3, size=fs, fw=weight, frame=False,
                            c=c)


labs = [
    "Inverted", "Inverted", "Transformed", "Transformed", "Transformed",
    "Assumed", "Calculated", "Calculated"
]

#add_labs_to_col(1, labs)

labs = [
    "Transformed", "Transformed", "Inverted", "Inverted", "Inverted",
    "Assumed", "Calculated", "Calculated"
]

geom = pg.load("geom_TL_t%s.bms" % timestep)
labs[-1] = "Inverted"

# Add labels for covariance reference
ax = grid.axes_all[0]
mid = geom.xmax() / 2

kwargs = dict(va="center", ha="center", fontsize=fs, fontweight="semibold")
ax.text(35, -8, "I", color="w", **kwargs)
ax.text(115, -8, "II", color="w", **kwargs)
ax.text(35, -20, "III", color="w", **kwargs)
ax.text(115, -20, "IV", color="w", **kwargs)

#add_labs_to_col(2, labs)

for i, (ax, label) in enumerate(zip(grid.axes_column[0], long_labels)):
    ax.set_yticks([-5, -15, -25])
    ax.set_yticklabels([" 5", "15", "25\n"])
    ax.set_ylabel("Depth (m)", labelpad=1)
    color = "k" if i not in (1, 3, 7) else "w"
    add_inner_title(ax, label, loc=3, c=color, frame=False)

    if i == 5:
        for bound in geom.boundaries():
            style = "solid"
            if np.isclose(bound.size(), 14.14, atol=0.5):
                style = "dotted"
            pg.mplviewer.drawSelectedMeshBoundaries(ax, [bound], linewidth=0.5,
                                                    linestyles=style,
                                                    color="k")        

for i, (ax, label) in enumerate(zip(grid.axes_column[0], long_labels)):
     pg.mplviewer.drawPLC(ax, geom, fillRegion=False, lw=0.5)

for i, (ax, label) in enumerate(zip(grid.axes_column[1], long_labels)):
     pg.mplviewer.drawPLC(ax, geom, fillRegion=False, lw=0.5)

for i, (ax, label) in enumerate(zip(grid.axes_column[2], long_labels)):
     pg.mplviewer.drawPLC(ax, geom, fillRegion=False, lw=0.5)


for i, ax in enumerate(grid.axes_all):
    ax.set_facecolor("0.45")
    ax.plot(sensors,
            np.zeros_like(sensors) + 0.1, marker="o", lw=0, color="k", ms=0.6)
    ax.tick_params(axis='both', which='major')
    ax.set_xticks([25, 50, 75, 100, 125])
    ax.set_ylim(-30, 0)
    ax.set_aspect(1.85)

for row in grid.axes_row[:-1]:
    for ax in row:
        ax.xaxis.set_visible(False)

for ax in grid.axes_row[-1]:
    ax.set_xlabel("x (m)", labelpad=0.2)

fig.savefig("TL_models.pdf", dpi=500, bbox_inches="tight", pad_inches=0.0)
