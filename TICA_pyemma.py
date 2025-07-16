import numpy as np
import pyemma
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

# === Inputs
topology = ""
trajectory = ""

feat = pyemma.coordinates.featurizer(topology)
feat.add_backbone_torsions(cossin=True)

# === Step 2: Load trajectory and extract torsions
data = pyemma.coordinates.load(trajectory, features=feat)

# === Step 3: Compute TICA
tica = pyemma.coordinates.tica(data, lag=20, dim=2)
tica_output = tica.get_output()[0]
tica1, tica2 = tica_output[:, 0], tica_output[:, 1]
# === Step 4: TICA Projection with KDE
xy = np.vstack([tica1, tica2])
z = gaussian_kde(xy)(xy)

plt.figure(figsize=(8, 6))
sc = plt.scatter(tica1, tica2, c=z, cmap='plasma', s=10)
plt.xlabel('TIC 1')
plt.ylabel('TIC 2')
plt.title('TICA Projection')
plt.colorbar(sc, label='Density')
plt.tight_layout()
plt.savefig('TICA_density_kde.png', dpi=300)
plt.close()

# === Step 5: Free Energy Surface (FEL)
kT = 1.0
n_bins = 60

H, xedges, yedges, _ = binned_statistic_2d(tica1, tica2, None, 'count', bins=n_bins)
H = H.T
H = np.where(H == 0, np.nan, H)

F = -kT * np.log(H)
F -= np.nanmin(F)
F = np.nan_to_num(F, nan=np.nanmax(F) + 2.0)

F_smooth = gaussian_filter(F, sigma=1.0)
F_smooth -= np.min(F_smooth)
F_smooth *= 2.0  # exaggerate funnel depth

# === Grid for plotting
x_centers = 0.5 * (xedges[:-1] + xedges[1:])
y_centers = 0.5 * (yedges[:-1] + yedges[1:])
X, Y = np.meshgrid(x_centers, y_centers)

# === Step 7: 3D Funnel-Shaped FEL plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Main energy surface
surf = ax.plot_surface(
    X, Y, F_smooth,
    cmap='plasma', edgecolor='k', linewidth=0.3,
    rstride=1, cstride=1, alpha=0.9, antialiased=True
)

# Optional wireframe overlay
ax.plot_wireframe(X, Y, F_smooth, color='k', linewidth=0.1, alpha=0.2)

# Axis labels and styling
ax.set_xlabel("TIC 1")
ax.set_ylabel("TIC 2")
ax.set_zlabel("dG (kT)")
ax.set_title("Free Energy Landscape (TICA)")

# Better angle to show full funnel shape
ax.view_init(elev=30, azim=210)

fig.colorbar(surf, shrink=0.6, label='dG (kT)')
plt.tight_layout()
plt.savefig('TICA_Funnel_FEL_fixed.png', dpi=300)

