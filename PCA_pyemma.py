import MDAnalysis as mda
from MDAnalysis.analysis.align import AlignTraj
from pyemma.coordinates import featurizer, source, pca
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import pyemma
from MDAnalysis.analysis import align


topology = ""
trajectory = ""


# load the trajectory
u = mda.Universe(topology, trajectory)

# load the reference PDB
ref = mda.Universe(topology)

# align trajectory on backbone to the reference PDB
aligner = align.AlignTraj(u, ref, select="backbone", in_memory=True)
aligner.run()

# now write out the aligned frames to disk
with mda.Writer("aligned.xtc", n_atoms=u.atoms.n_atoms) as W:
    for ts in u.trajectory:
        W.write(u.atoms)

# === Step 3: PyEMMA feature extraction on aligned trajectory
feat = pyemma.coordinates.featurizer(topology)
feat.add_backbone_torsions(cossin=True)

data_src = pyemma.coordinates.source("aligned.xtc", features=feat, top=topology)

# === Step 4: Run PCA
pca_model = pyemma.coordinates.pca(data_src, dim=2)
Y = pca_model.get_output()[0]
pc1, pc2 = Y[:, 0], Y[:, 1]

print("PC1 range:", pc1.min(), pc1.max())
print("PC2 range:", pc2.min(), pc2.max())

# === Step 5: Density scatter
xy = np.vstack([pc1, pc2])
kde = gaussian_kde(xy)

# grid for density
xmin, xmax = pc1.min(), pc1.max()
ymin, ymax = pc2.min(), pc2.max()
Xgrid, Ygrid = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
grid_coords = np.vstack([Xgrid.ravel(), Ygrid.ravel()])
Z = kde(grid_coords).reshape(Xgrid.shape)
Z_smooth = gaussian_filter(Z, sigma=1.0)

# color by density at each point
point_density = kde(xy)

plt.figure(figsize=(8, 6))
sc = plt.scatter(pc1, pc2, c=point_density, cmap='viridis', s=10)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection ")
plt.colorbar(sc, label="Density")
plt.tight_layout()
plt.savefig("PCA_aligned_scatter.png", dpi=300)



# Parameters
kT = 1.0
n_bins = 200

# 2D histogram of PC1, PC2
H, xedges, yedges = np.histogram2d(pc1, pc2, bins=n_bins)
H = H.T

# Bin area
dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
A = dx * dy
N_total = np.sum(H)

# Replace zeros
H[H == 0] = 1e-8

# Free energy (GROMACS logic)
P = H / (N_total * A)
F = -kT * np.log(P)
F -= np.min(F)
F_smooth = gaussian_filter(F, sigma=1.2)

# Grid
xcenters = 0.5 * (xedges[:-1] + xedges[1:])
ycenters = 0.5 * (yedges[:-1] + yedges[1:])
X, Y = np.meshgrid(xcenters, ycenters)

# === Plot ===
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

# Surface with soft colors
surf = ax.plot_surface(
    X, Y, F_smooth,
    cmap='viridis',
    edgecolor='none',      # No black edge
    linewidth=0,
    rstride=1,
    cstride=1,
    alpha=0.5,             
    antialiased=True
)

# Wireframe with soft color (light gray or muted blue)
ax.plot_wireframe(
    X, Y, F_smooth,
    color='#cccccc',       
    linewidth=0.2,
    alpha=0.5
)

# Labels and viewing
ax.set_xlabel("PC1", labelpad=10)
ax.set_ylabel("PC2", labelpad=10)
ax.set_zlabel("dG (kT)", labelpad=10)
ax.set_title("Free Energy Landscape (PCA)", pad=15)

ax.view_init(elev=25, azim=230)
ax.set_zlim(-1, np.max(F_smooth))
ax.set_box_aspect([2, 2, 1.5]) 

# Soft colorbar
cbar = fig.colorbar(surf, shrink=0.6, pad=0.1)
cbar.set_label("dG (kT)")

# Save
plt.tight_layout()
plt.savefig("PCA_FEL.png", dpi=300)

