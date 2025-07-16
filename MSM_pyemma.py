import pyemma
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyemma.coordinates import cluster_kmeans

# === Inputs ===
topology   = ""
trajectory = ""
lag_time   = 20
n_tica_dim = 2
n_clusters = 100
n_macrostates = 5

# === Step 1: Feature extraction ===
feat = pyemma.coordinates.featurizer(topology)
feat.add_backbone_torsions(cossin=True)

# === Step 2: Load trajectory features ===
data = pyemma.coordinates.load(trajectory, features=feat)

# === Step 3: TICA ===
tica_model   = pyemma.coordinates.tica(data, lag=lag_time, dim=n_tica_dim)
tica_output  = tica_model.get_output()[0]           # shape (n_frames, n_tica_dim)
tica1, tica2 = tica_output[:, 0], tica_output[:, 1]

# === Step 4: Clustering into microstates ===
kmeans = cluster_kmeans(tica_output, k=n_clusters, max_iter=1000, stride=1)
dtrajs = kmeans.dtrajs

# === Step 5: MSM estimation ===
msm = pyemma.msm.estimate_markov_model(dtrajs, lag=lag_time)

# === Plot 1: MSM Transition Matrix ===
plt.figure(figsize=(10, 8))
sns.heatmap(msm.transition_matrix, cmap="Blues", square=True)
plt.title("MSM Transition Matrix (Microstates)")
plt.xlabel("State i")
plt.ylabel("State j")
plt.tight_layout()
plt.savefig("02_Transition_Matrix.png", dpi=300)
plt.clf()

# === Plot 2: Stationary Distribution ===
pi = msm.stationary_distribution
plt.figure(figsize=(8, 6))
plt.bar(range(len(pi)), pi)
plt.title("Stationary Distribution of Microstates")
plt.xlabel("Microstate Index")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig("03_Stationary_Distribution.png", dpi=300)
plt.clf()

# === Plot 3: MFPT Matrix ===
n_states = msm.nstates
mfpt_matrix = np.zeros((n_states, n_states))
for i in range(n_states):
    for j in range(n_states):
        if i != j:
            mfpt_matrix[i, j] = msm.mfpt(i, j)

plt.figure(figsize=(10, 8))
sns.heatmap(mfpt_matrix, cmap="magma_r", square=True)
plt.title("MFPT Matrix")
plt.xlabel("Target State")
plt.ylabel("Start State")
plt.tight_layout()
plt.savefig("04_MFPT_Matrix.png", dpi=300)
plt.clf()

# === Step 6: PCCA macrostates ===
pcca = msm.pcca(n_macrostates)
macro_assignments = pcca.assignments
macro_assignment_full = np.full(n_clusters, -1)
macro_assignment_full[msm.active_set] = macro_assignments
frame_macrostate = np.array([macro_assignment_full[m] for m in dtrajs[0]])

# === Step 7: Combined PCCA scatter + network ===
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Plot PCCA scatter using tab10
pcca_colors = plt.get_cmap("tab10")(np.arange(n_macrostates))
for i in range(n_macrostates):
    idx = np.where(frame_macrostate == i)[0]
    ax.scatter(
        tica_output[idx, 0],
        tica_output[idx, 1],
        s=8,
        color=pcca_colors[i],
        alpha=0.4,
        label=f"Macrostate {i+1}"
    )

# Calculate macrostate centers
macro_pos = {}
for i in range(n_macrostates):
    idx = np.where(frame_macrostate == i)[0]
    macro_pos[i] = (
        np.mean(tica_output[idx, 0]),
        np.mean(tica_output[idx, 1])
    )

# Build macrostate network
G = nx.DiGraph()
macro_transmat = pcca.coarse_grained_transition_matrix
for i in range(n_macrostates):
    propensity = pi[pcca.metastable_sets[i]].sum()
    G.add_node(i, size=propensity * 3000)

threshold = 0.01
for i in range(n_macrostates):
    for j in range(n_macrostates):
        if i != j and macro_transmat[i, j] > threshold:
            G.add_edge(i, j, weight=macro_transmat[i, j])

# Use different colormap for network nodes (Set2)
network_colors = plt.get_cmap("Set2")(np.linspace(0, 1, n_macrostates))

# Draw network nodes
nx.draw_networkx_nodes(
    G, pos=macro_pos,
    node_size=[G.nodes[n]['size'] for n in G.nodes],
    node_color=network_colors,
    edgecolors='black',
    linewidths=1.2,
    alpha=0.95,
    ax=ax
)

# Draw edges
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw_networkx_edges(
    G, pos=macro_pos,
    edgelist=G.edges(),
    width=[3 + 10 * w for w in edge_weights],
    alpha=0.7,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=20,
    ax=ax
)

# Edge labels
edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(
    G, pos=macro_pos,
    edge_labels=edge_labels,
    font_size=10,
    ax=ax
)

# Node labels (1-based indexing)
nx.draw_networkx_labels(
    G, pos=macro_pos,
    labels={n: f"{n+1}" for n in G.nodes},
    font_color="black",
    font_size=12,
    font_weight="bold",
    ax=ax
)

# Final touches
ax.set_xlabel("TIC1")
ax.set_ylabel("TIC2")
ax.set_title("PCCA Macrostates and MSM Network")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("05_PCCA_Macrostate_TICA_Network.png", dpi=300)
plt.clf()

print(" All plots saved:")
print(" 02_Transition_Matrix.png")
print(" 03_Stationary_Distribution.png")
print(" 04_MFPT_Matrix.png")
print(" 05_PCCA_Macrostate_TICA_Network.png")

