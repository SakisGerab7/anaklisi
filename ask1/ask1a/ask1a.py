from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# =========================================
# Δημιουργία μη κατευθυνόμενου γράφου
# (όλα τα links είναι bidirectional)
# =========================================

G = nx.Graph()

# Κόμβοι
nodes = list(range(1, 15))
G.add_nodes_from(nodes)

# Ακμές από το σχήμα
edges = [

    # Αριστερό cluster
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),

    (2, 3),
    (2, 5),
    (3, 4),

    (3, 6),

    (4, 5),
    (4, 7),

    (5, 10),

    # Κάτω cluster
    (6, 7),
    (6, 8),
    (6, 9),

    (7, 9),

    (8, 9),

    # Δεξί cluster
    (10, 12),
    (10, 11),
    (10, 14),

    (11, 13),
    (11, 14),

    (12, 13),
    (12, 14),

    (13, 14)
]

G.add_edges_from(edges)

# =========================================
# Personalization vector
# 50% στον κόμβο 14
# Το υπόλοιπο 50% ισομερώς στους άλλους
# =========================================

personalization = {}

for n in nodes:
    if n == 14:
        personalization[n] = 0.5
    else:
        personalization[n] = 0.5 / (len(nodes) - 1)

# =========================================
# Υπολογισμός Personalized PageRank
# =========================================

alpha = 0.65

pagerank = nx.pagerank(
    G,
    alpha=alpha,
    personalization=personalization
)

# =========================================
# Ταξινόμηση κόμβων
# =========================================

sorted_pr = sorted(
    pagerank.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\nPageRank scores:\n")

for node, score in sorted_pr:
    print(f"Node {node}: {score:.6f}")

# =========================================
# Σχεδίαση γράφου
# =========================================

fig, ax = plt.subplots(figsize=(9, 6))

pos = nx.spring_layout(G, seed=9)

max_pagerank = max(pagerank.values())
min_pagerank = min(pagerank.values())

normalized_pagerank = {node: (pagerank[node] - min_pagerank) / (max_pagerank - min_pagerank) for node in G.nodes()}

red_intensity = [int(255 * (0.5 + 0.5 * np.cos((normalized_pagerank[node] - 1.0) * 4 * np.pi / 3))) for node in G.nodes()]
green_intensity = [int(255 * (0.5 + 0.5 * np.cos((normalized_pagerank[node] - 0.5) * 4 * np.pi / 3))) for node in G.nodes()]
blue_intensity = [int(255 * (0.5 + 0.5 * np.cos((normalized_pagerank[node] - 0.0) * 4 * np.pi / 3))) for node in G.nodes()]

x = np.linspace(0, 1, 256)

r = 0.5 + 0.5 * np.cos((x - 1.0) * 4 * np.pi / 3)
g = 0.5 + 0.5 * np.cos((x - 0.5) * 4 * np.pi / 3)
b = 0.5 + 0.5 * np.cos((x - 0.0) * 4 * np.pi / 3)

colors = np.column_stack((r, g, b))

pagerank_cmap = ListedColormap(colors)

node_colors = [
    f'#{r:02x}{g:02x}{b:02x}'
    for r, g, b in zip(
        red_intensity,
        green_intensity,
        blue_intensity
    )
]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_colors,
    node_size=[
        500 + 2000*np.sqrt(normalized_pagerank[node])
        for node in G.nodes()
    ],
    font_color='white',
    arrows=False
)

pr_values = list(normalized_pagerank.values())

vmin = min(pr_values)
vmax = max(pr_values)

norm = Normalize(vmin=vmin, vmax=vmax)

sm = ScalarMappable(
    cmap=pagerank_cmap,
    norm=norm
)

sm.set_array([])

# Horizontal colorbar with custom ticks and labels
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0], labels=[f'{min_pagerank:.4f}', f'{min_pagerank + 0.25*(max_pagerank - min_pagerank):.4f}', f'{min_pagerank + 0.5*(max_pagerank - min_pagerank):.4f}', f'{min_pagerank + 0.75*(max_pagerank - min_pagerank):.4f}', f'{max_pagerank:.4f}'])

plt.tight_layout()
plt.savefig(f"pagerank_graph.png")
plt.show()

# =========================================
# Bar plot PageRank
# =========================================

plt.figure(figsize=(10, 5))

nodes_sorted = [x[0] for x in sorted_pr]
scores_sorted = [x[1] for x in sorted_pr]

plt.bar(
    [str(n) for n in nodes_sorted],
    scores_sorted
)

plt.xlabel("Node")
plt.ylabel("PageRank")
plt.title("Personalized PageRank Scores")
plt.grid(axis='y')
plt.savefig("pagerank_bar.png")
plt.show()