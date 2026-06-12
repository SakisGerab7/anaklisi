import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from scipy.stats import kendalltau
from itertools import combinations

# Δημιουργία κατευθυνόμενου γράφου
G = nx.DiGraph()

# Κόμβοι
nodes = list(range(8))
G.add_nodes_from(nodes)

# Ακμές από το σχήμα
edges = [
    (0, 1),
    (0, 6),
    (0, 7),

    (1, 2),
    (1, 7),

    (2, 1),
    (2, 7),

    (3, 5),
    (3, 7),

    (4, 5),

    (5, 6),

    (6, 5),

    (7, 6)
]

G.add_edges_from(edges)

# Τιμές damping factor
alphas = [0.55, 0.65, 0.75, 0.85, 0.95]

# Υπολογισμός PageRank για κάθε alpha
pageranks = {}

for a in alphas:
    try:
        pr = nx.pagerank(G, alpha=a, max_iter=1000)
    except:
        pass
    finally:
        pageranks[a] = pr

# ==========================
# Σχεδίαση μεταβολής PageRank
# ==========================

plt.figure(figsize=(10, 6))

for node in nodes:
    y = [pageranks[a][node] for a in alphas]
    plt.plot(alphas, y, marker='o', label=f'Node {node}')

plt.xlabel("Damping Factor α")
plt.ylabel("PageRank")
plt.title("Variation of PageRank for different damping factors")
plt.legend()
plt.grid(True)
plt.show()

# ==================================
# 8 ξεχωριστά διαγράμματα (ένα/node)
# ==================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, node in enumerate(nodes):
    y = [pageranks[a][node] for a in alphas]

    axes[i].plot(alphas, y, marker='o')
    axes[i].set_title(f'Node {node}')
    axes[i].set_xlabel('α')
    axes[i].set_ylabel('PageRank')
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# ==================================
# Rankings για κάθε damping factor
# ==================================

rankings = {}

for a in alphas:
    sorted_nodes = sorted(
        pageranks[a],
        key=pageranks[a].get,
        reverse=True
    )
    rankings[a] = sorted_nodes

    print(f"\nRanking for a = {a}")
    print(sorted_nodes)

# ==================================
# Kendall Tau αποστάσεις
# ==================================

print("\nKendall Tau between rankings:\n")

for a1, a2 in combinations(alphas, 2):

    r1 = rankings[a1]
    r2 = rankings[a2]

    # Μετατροπή ranking σε θέση
    pos1 = [r1.index(n) for n in nodes]
    pos2 = [r2.index(n) for n in nodes]

    tau, p = kendalltau(pos1, pos2)

    print(f"a={a1} vs a={a2}")
    print(f"Kendall tau = {tau:.4f}")
    print("-" * 30)

# ==================================
# Σχεδίαση γράφου
# ==================================
for alpha in alphas:
    fig, ax = plt.subplots(figsize=(9, 6))

    positions = nx.spring_layout(G, seed=43)  # Θέσεις κόμβων με το spring layout

    max_pagerank = max(pageranks[alpha].values())
    min_pagerank = min(pageranks[alpha].values())

    normalized_pagerank = {node: (pageranks[alpha][node] - min_pagerank) / (max_pagerank - min_pagerank) for node in G.nodes()}

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
        positions,
        with_labels=True,
        node_color=node_colors,
        node_size=[
            1000 + 4000*np.sqrt(normalized_pagerank[node])
            for node in G.nodes()
        ],
        font_color='white',
        arrows=True
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

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0], labels=[f'{min_pagerank:.4f}', f'{min_pagerank + 0.25*(max_pagerank - min_pagerank):.4f}', f'{min_pagerank + 0.5*(max_pagerank - min_pagerank):.4f}', f'{min_pagerank + 0.75*(max_pagerank - min_pagerank):.4f}', f'{max_pagerank:.4f}'])

    plt.title(f"Directed Graph (α={alpha})")
    plt.tight_layout()
    plt.savefig(f"graph_alpha_{alpha}.png")
    plt.show()

# plt.figure(figsize=(8, 6))

# pos = nx.spring_layout(G, seed=43)  # Θέσεις κόμβων με το spring layout

# max_pagerank = max(pageranks[0.55].values())
# min_pagerank = min(pageranks[0.55].values())

# normalized_pagerank = {node: (pageranks[0.55][node] - min_pagerank) / (max_pagerank - min_pagerank) for node in G.nodes()}

# red_intensity = [int(255 * (0.5 + 0.5 * np.cos((normalized_pagerank[node] - 1.0) * 4 * np.pi / 3))) for node in G.nodes()]
# green_intensity = [int(255 * (0.5 + 0.5 * np.cos((normalized_pagerank[node] - 0.5) * 4 * np.pi / 3))) for node in G.nodes()]
# blue_intensity = [int(255 * (0.5 + 0.5 * np.cos((normalized_pagerank[node] - 0.0) * 4 * np.pi / 3))) for node in G.nodes()]

# nx.draw(
#     G,
#     pos,
#     with_labels=True,
#     node_color=[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in zip(red_intensity, green_intensity, blue_intensity)],
#     node_size=[1000 + 4000 * np.sqrt(normalized_pagerank[node]) for node in G.nodes()],
#     font_color='white',
#     arrows=True
# )

# plt.title("Directed Graph")
# plt.show()