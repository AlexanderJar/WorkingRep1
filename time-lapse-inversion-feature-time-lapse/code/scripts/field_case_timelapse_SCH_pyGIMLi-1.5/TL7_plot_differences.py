from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import ImageGrid

import pygimli as pg
from fpinv import add_inner_title, logFormat, rst_cov, set_style
from pygimli.viewer.mpl import drawModel
#from settings import plot_boreholes

fs = 5.5
set_style(fs, style="seaborn-v0_8-dark")

ntimes = 3

mesh = pg.load("paraDomain_1_t1.bms")
sensors = np.loadtxt("sensors_t1.npy")

def to_sat(fw, fi, fa, fr):
    phi = 1 - fr
    return fw / phi, fi / phi, fa / phi

fij_t = np.zeros((mesh.cellCount(),ntimes))


for timestep in range(1,ntimes+1):

    # Load data
    joint = np.load("TL_joint_inversion_SCH_lam10.0_zW0.25_delta100.0_gam0_eps0_it10_new_gs.npz")

    # Joint inversion no TL
    veljoint1, rhojoint1, faj1, fij1, fwj1, frj1, maskj1 = joint["vel"], joint[
        "rho"], joint["fa"], joint["fi"], joint["fw"], joint["fr"], joint["mask"]

    nCellsj = mesh.cellCount()

    veljoint1 = veljoint1[nCellsj*(timestep-1):nCellsj*timestep]
    rhojoint1 = rhojoint1[nCellsj*(timestep-1):nCellsj*timestep]
    faj1 = faj1[nCellsj*(timestep-1):nCellsj*timestep]
    fij1 = fij1[nCellsj*(timestep-1):nCellsj*timestep]
    fwj1 = fwj1[nCellsj*(timestep-1):nCellsj*timestep]
    frj1 = frj1[nCellsj*(timestep-1):nCellsj*timestep]
    maskj1 = maskj1[nCellsj*(timestep-1):nCellsj*timestep]

    sumj1 = faj1 + fij1 + fwj1 + frj1
    
    # Rock and ice
    rij1 = fij1 + frj1

    #fwj1, fij1, faj1 = to_sat(fwj1, fij1, faj1, frj1)

    fij_t[:,timestep-1] = fij1[nCellsj*(timestep-1):nCellsj*timestep]

    
    
    # Some helper functions
    def update_ticks(cb, label="", logScale=False, cMin=None, cMax=None):
        cb.set_ticks([cMin, cMax])
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
        if logScale:
            for lab in cb.ax.yaxis.get_minorticklabels():
                lab.set_visible(False)


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
        if not np.isclose(model.min(), 0.0, atol=9e-3) and (model < 0).any():
            model = np.ma.masked_where(model < 0, model)
            model = np.ma.masked_where(model > 1, model)

        if "coverage" in kwargs:
            model = np.ma.masked_where(kwargs["coverage"] == 0, model)
        gci = drawModel(ax, mesh, model, rasterized=True, nLevs=2, **kwargs)
        return gci


    def minmax(data):
        """Return minimum and maximum of data as a 2-line string."""
        tmp = np.array(data)
        print("max", tmp.max())
        if np.isclose(tmp.min(), 0, atol=9e-3):
            min = 0
        else:
            min = tmp.min()
        if np.max(tmp) > 10 and np.max(tmp) < 1e4:
            return "min: %d | max: %d" % (min, tmp.max())
        if np.max(tmp) > 1e4:
            return "min: %d" % min + " | max: " + logFormat(tmp.max())
        else:
            return "min: %.2f | max: %.2f" % (min, tmp.max())


    # %%
    fig = plt.figure(figsize=(10, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(8, 2), axes_pad=[0.03, 0.03],
                     share_all=True, cbar_location="right",                    #add_all=True,
                     cbar_mode="edge", cbar_size="5%", cbar_pad=0.05, aspect=True)

    cov = rst_cov(mesh, np.loadtxt("rst_coverage_t%s.dat" % timestep))

    labels = ["v (m/s)", r" $\rho$ ($\Omega$m)"]
    labels.extend([r"f$_{\rm %s}$" % x for x in "wiar"])
    labels.extend([r"$\Sigma$ f$_{\rm k}$", r"f$_{\rm i}$ + f$_{\rm r}$"])

    long_labels = [
        "Velocity", "Resistivity", "Water content", "Ice content", "Air content",
        "Rock content", "Sum all", "Sum ice and rock"
    ]
    meshs = [mesh, mesh]
    cmaps = ["viridis", "Spectral_r", "Blues", "Purples", "Greens", "Oranges",  "RdBu_r","RdPu"]
    datas = [(veljoint1, veljoint2), (rhojoint1, rhojoint2),
         (fwj1, fwj2), (fij1, fij2), (faj1, faj2),
         (frj1, frj2), (sumj1, sumj2),(rij1, rij2)]

    for i, (row, data, label,
            cmap) in enumerate(zip(grid.axes_row, datas, labels, cmaps)):
        print("Plotting", label)
        if i == 0:
            lims = {"cMin": 500, "cMax": 3500}
        elif i == 1:
            lims = {"cMin": 500, "cMax": 10000}
        elif i == 2:  # water
            lims = {"cMin": 0.01, "cMax": 0.7}
        elif i == 3:  # ice
            lims = {"cMin": 0.01, "cMax": 0.5}
        elif i == 4:  # air
            lims = {"cMin": 0.01, "cMax": 0.8}
        elif i == 5:  # rock
            lims = {"cMin": 0.3, "cMax": 0.5}
        elif i == 6:
            lims = {"cMin": 0.95, "cMax": 1.05}
        elif i == 7:
            lims = {"cMin": 0.01, "cMax": 1.0}
        else:
            lims = lim(list(data[0][cov > 0]) + list(data[1][cov > 0]))
        logScale = True if "rho" in label else False
        ims = []
        for j, ax in enumerate(row):
            if data[j] is None:
                ims.append(None)
                continue
            coverage = cov
            pg.boxprint(j)
            #print(meshs[j], len(coverage), len(data[j]))
            #color = "k" if j is 0 and i not in (1, 3, 5) else "w"
            ims.append(
                draw(ax, meshs[j], data[j], **lims, logScale=logScale,
                     coverage=coverage))
            ax.text(0.987, 0.05, minmax(data[j][coverage > 0]),
                    transform=ax.transAxes, fontsize=fs, ha="right", color="w")
            ims[j].set_cmap(cmap)

        cb = fig.colorbar(ims[1], cax=grid.cbar_axes[i])
        update_ticks(cb, label=label, logScale=logScale, **lims)

    for ax, title, num in zip(grid.axes_row[0], [
            "PJI",
            "TLPJI"
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
        "Transformed", "Transformed", "Inverted", "Inverted", "Inverted",
        "Inverted", "Calculated", "Calculated"
    ]

    add_labs_to_col(0, labs)
    add_labs_to_col(1, labs)


    # Show box around ice constraint
    #box = pg.load("box.bms")
    #ax = grid.axes_column[2][3]
    #pg.mplviewer.drawMeshBoundaries(ax, box, fitView=False, lw=0.5)

    for i, ax in enumerate(grid.axes_all):
        ax.set_facecolor("0.45")
        ax.plot(sensors[:52, 0], sensors[:52, 1] + 0.1, marker="o", lw=0, color="k",
                ms=0.6)
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(-20, 0.8)
        ax.set_xlim(-5, 55)
        if i < 3:
            c = "w"
        else:
            c = "k"
        #plot_boreholes(ax, lw=0.5, color=c)

    for row in grid.axes_row[:-1]:
        for ax in row:
            ax.xaxis.set_visible(False)

    for ax in grid.axes_column[-1]:
        ax.yaxis.set_visible(False)

    for ax in grid.axes_row[-1]:
        ax.set_xlabel("x (m)", labelpad=0.2)

    for i, (ax, label) in enumerate(zip(grid.axes_column[0], long_labels)):
        ax.set_yticks([-0, -5, -10, -15])
        ax.set_yticklabels([" 0", " 5", "10", "15"])
        ax.set_ylabel("Depth (m)", labelpad=1)
        # add_inner_title(ax, label, loc=2, c="k", frame=False)

    ax = grid.axes_column[0][5]

    #ax.annotate("SCH_5198", xy=(10.5, -4.5), xycoords='data', xytext=(5, 0),
    #            textcoords='offset points', va="center", ha="left",
    #            arrowprops=dict(arrowstyle="->", lw=0.5))

    #ax.annotate("SCH_5000", xy=(26.5, -6.6), xycoords='data', xytext=(5, 0),
    #            textcoords='offset points', va="center", ha="left",
    #            arrowprops=dict(arrowstyle="->", lw=0.5))

    fig.savefig("TL_joint_inversion_SCH_lam10.0_zW0.25_delta0vs100_gam0_eps0.0_it10_poros_t%s.pdf" % timestep, dpi=500, bbox_inches="tight",
                pad_inches=0.0)
    # fig.savefig("4PM_joint_inversion.png", dpi=150, bbox_inches="tight")
