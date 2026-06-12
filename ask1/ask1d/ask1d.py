import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# ΑΡΧΙΚΟΣ ΓΡΑΦΟΣ
# ============================================

G = nx.Graph()

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

# ============================================
# GOOGLE MATRIX
# ============================================

alpha = 0.85

google_matrix = nx.google_matrix(G, alpha=alpha)

# Υπολογισμός ιδιοτιμών
eigenvalues_original = np.linalg.eigvals(google_matrix)

print("\nEigenvalues of the original graph:\n")

for ev in eigenvalues_original:
    print(f"{ev:.6f}")

# ============================================
# ΝΕΕΣ ΑΚΜΕΣ
# ============================================

new_edges = [
    (5, 11),
    (4, 11),
    (7, 13),
    (8, 12),
    (9, 13)
]

G2 = G.copy()
G2.add_edges_from(new_edges)

# ============================================
# GOOGLE MATRIX ΝΕΟΥ ΓΡΑΦΟΥ
# ============================================

google_matrix_new = nx.google_matrix(G2, alpha=alpha)

# Νέες ιδιοτιμές
eigenvalues_new = np.linalg.eigvals(google_matrix_new)

print("\nEigenvalues of the new graph:\n")

for ev in eigenvalues_new:
    print(f"{ev:.6f}")

# ============================================
# ΣΧΕΔΙΑΣΗ ΣΥΓΚΡΙΣΗΣ
# ============================================

plt.figure(figsize=(10, 6))

# Bar plots for original and new eigenvalues
width = 0.35
x = np.arange(len(eigenvalues_original))
plt.bar(x - width/2, eigenvalues_original, width=width, label='Original Graph')
plt.bar(x + width/2, eigenvalues_new, width=width, label='New Graph')
plt.xlabel('Eigenvalues (λ)')
plt.ylabel('Magnitude |λ|')
plt.title('Comparison of Eigenvalues in Google Matrix')
plt.xticks(x, [f'λ{i+1}' for i in range(len(eigenvalues_original))])
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('eigenvalues_comparison.png')  # Save the figure
plt.show()
