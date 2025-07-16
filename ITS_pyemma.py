import pyemma
import numpy as np
import matplotlib.pyplot as plt
import pyemma.plots as mplt
from pyemma.coordinates import cluster_kmeans
from pyemma.msm import its


def main():
    # === Inputs
    topology = ""
    trajectory = ""

    # Feature extraction
    feat = pyemma.coordinates.featurizer(topology)
    feat.add_backbone_torsions(cossin=True)

    # Load trajectory
    data = pyemma.coordinates.load(trajectory, features=feat)

    # TICA
    tica_model = pyemma.coordinates.tica(data, lag=20, dim=2)
    tica_output = tica_model.get_output()[0]
    np.savetxt("tica_output.txt", tica_output)

    # Clustering and ITS
    for k in [100]:
        kmeans = cluster_kmeans(tica_output, k=k, max_iter=1000, stride=1)
        dtrajs = kmeans.dtrajs

        
        its_result = its(dtrajs, lags=50, nits=5)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        mplt.plot_implied_timescales(its_result, ax=ax, units='steps')

        # Get Line2D handles from the axes
        handles = ax.get_lines()
        labels = [f"Timescale {i+1}" for i in range(len(handles))]

        ax.legend(handles, labels, loc='upper left')
        ax.set_title(f"Implied timescales (ITS)")
        ax.set_xlabel("Lag time")
        ax.set_ylabel("Timescale")
        plt.tight_layout()
        plt.savefig(f"ITS_k{k}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
