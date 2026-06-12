import numpy as np
import networkx as nx

# =========================================
# Δημιουργία directed graph
# =========================================

edges = [
    (3, 8),
    (3, 10),

    (5, 11),

    (7, 8),
    (7, 11),

    (8, 9),

    (11, 2),
    (11, 9),
    (11, 10)
]

G = nx.DiGraph()
G.add_edges_from(edges)

# =========================================
# Κόμβοι
# =========================================

nodes = sorted(G.nodes())
n = len(nodes)

node_index = {node: i for i, node in enumerate(nodes)}

print("Nodes:", nodes)

# =========================================
# Adjacency Matrix A
# =========================================

A = nx.to_numpy_array(G, nodelist=nodes)

print("\nAdjacency Matrix A:\n")
print(A)

# =========================================
# Authorities Matrix
# A^T A
# =========================================

ATA = A.T @ A

print("\nAuthorities Matrix (A^T A):\n")
print(ATA)

# =========================================
# Hubs Matrix
# A A^T
# =========================================

AAT = A @ A.T

print("\nHubs Matrix (A A^T):\n")
print(AAT)

# =========================================
# Eigenvalues / Eigenvectors Authorities
# =========================================

eigvals_auth, eigvecs_auth = np.linalg.eig(ATA)

print("\nAuthorities Eigenvalues:\n")
print(eigvals_auth)

print("\nAuthorities Eigenvectors:\n")
print(eigvecs_auth)

# =========================================
# Eigenvalues / Eigenvectors Hubs
# =========================================

eigvals_hub, eigvecs_hub = np.linalg.eig(AAT)

print("\nHubs Eigenvalues:\n")
print(eigvals_hub)

print("\nHubs Eigenvectors:\n")
print(eigvecs_hub)

# =========================================
# Κυρίαρχα eigenvectors (HITS scores)
# =========================================

# Authorities
max_idx_auth = np.argmax(eigvals_auth.real)
principal_auth = eigvecs_auth[:, max_idx_auth].real

# Hubs
max_idx_hub = np.argmax(eigvals_hub.real)
principal_hub = eigvecs_hub[:, max_idx_hub].real

# Κανονικοποίηση
principal_auth = principal_auth / np.linalg.norm(principal_auth)
principal_hub = principal_hub / np.linalg.norm(principal_hub)

print("\n=================================")
print("Principal Authority Eigenvector")
print("=================================")

for node, score in zip(nodes, principal_auth):
    print(f"Node {node}: {score:.6f}")

print("\n=================================")
print("Principal Hub Eigenvector")
print("=================================")

for node, score in zip(nodes, principal_hub):
    print(f"Node {node}: {score:.6f}")