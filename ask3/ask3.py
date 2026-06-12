import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from implicit.bpr import BayesianPersonalizedRanking

# ============================================================
# CONFIG
# ============================================================

LASTFM_PATH = "hetrec2011-lastfm-2k/user_artists.dat"
MOVIELENS_PATH = "ml-1m/ratings.dat"

RANDOM_STATE = 42

# ============================================================
# DATA LOADING
# ============================================================

def load_lastfm(path):
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"userID": "user", "artistID": "item"})
    df["value"] = 1
    return df[["user", "item", "value"]]


def load_movielens(path):
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user", "item", "rating", "timestamp"]
    )
    df["value"] = 1
    return df[["user", "item", "value"]]

# ============================================================
# SPLIT
# ============================================================

def train_test_split_userwise(df, test_ratio=0.2):
    rng = np.random.default_rng(RANDOM_STATE)

    train_rows, test_rows = [], []

    for user, group in df.groupby("user"):
        if len(group) < 5:
            continue

        idx = np.arange(len(group))
        rng.shuffle(idx)

        split = int(len(group) * (1 - test_ratio))

        train_rows.append(group.iloc[idx[:split]])
        test_rows.append(group.iloc[idx[split:]])

    return pd.concat(train_rows), pd.concat(test_rows)

# ============================================================
# MATRIX
# ============================================================

def create_matrix(df):
    users = sorted(df.user.unique())
    items = sorted(df.item.unique())

    user_map = {u: i for i, u in enumerate(users)}
    item_map = {i: j for j, i in enumerate(items)}

    rows = df.user.map(user_map)
    cols = df.item.map(item_map)

    mat = sp.csr_matrix(
        (np.ones(len(df)), (rows, cols)),
        shape=(len(users), len(items))
    )

    return mat, user_map, item_map

# ============================================================
# TEST DICT
# ============================================================

def build_test_dict(test_df, user_map, item_map):
    test_dict = {}

    for user, group in test_df.groupby("user"):
        if user not in user_map:
            continue

        uid = user_map[user]
        items = {item_map[i] for i in group.item if i in item_map}

        test_dict[uid] = items

    return test_dict

# ============================================================
# METRICS
# ============================================================

def precision_recall_at_k(model, train_matrix, test_dict, k):
    precisions, recalls = [], []

    for user in test_dict:
        relevant = test_dict[user]
        if len(relevant) == 0:
            continue

        recs, _ = model.recommend(
            userid=user,
            user_items=train_matrix[user],
            N=k,
            filter_already_liked_items=True
        )

        recs = set(recs)

        hits = len(recs & relevant)

        precisions.append(hits / k)
        recalls.append(hits / len(relevant))

    return np.mean(precisions), np.mean(recalls)

# ============================================================
# PARALLEL CORE
# ============================================================

def single_run(model_fn, train_matrix, test_dict, k):
    model = model_fn()
    train_matrix_local = train_matrix.copy()
    model.fit(train_matrix_local)
    return precision_recall_at_k(model, train_matrix_local, test_dict, k)

def parallel_evaluate(model_fn, train_matrix, test_dict, k, n_runs=10, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(single_run)(model_fn, train_matrix, test_dict, k)
        for _ in range(n_runs)
    )

    p = np.array([r[0] for r in results])
    r = np.array([r[1] for r in results])

    return p, r


def evaluate_mean_std(model_fn, train_matrix, test_dict, k, n_runs=10, n_jobs=-1):
    p, r = parallel_evaluate(model_fn, train_matrix, test_dict, k, n_runs, n_jobs)

    return (
        np.mean(p), np.std(p),
        np.mean(r), np.std(r)
    )

# ============================================================
# PLOTTING
# ============================================================

def plot_mean_std(x, mean, std, xlabel, ylabel, title, filename):
    mean = np.array(mean)
    std = np.array(std)

    plt.figure(figsize=(8,5))

    plt.plot(x, mean, marker="o")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ============================================================
# EXPERIMENT A
# ============================================================

def experiment_factors(train_matrix, test_dict, n_runs=10, n_jobs=-1):

    factors_range = list(range(10, 101, 10))

    p_mean, p_std = [], []
    r_mean, r_std = [], []

    for f in factors_range:

        model_fn = lambda: BayesianPersonalizedRanking(
            factors=f,
            learning_rate=0.01,
            regularization=0.01,
            iterations=100,
            random_state=None
        )

        pm, ps, rm, rs = evaluate_mean_std(
            model_fn,
            train_matrix,
            test_dict,
            k=10,
            n_runs=n_runs,
            n_jobs=n_jobs
        )

        p_mean.append(pm)
        p_std.append(ps)
        r_mean.append(rm)
        r_std.append(rs)

        print(f"F={f} | P={pm:.4f}±{ps:.4f} | R={rm:.4f}±{rs:.4f}")

    return factors_range, p_mean, p_std, r_mean, r_std

# ============================================================
# EXPERIMENT B
# ============================================================

def experiment_k(train_matrix, test_dict, n_runs=10, n_jobs=-1):

    ks = list(range(2, 21, 2))

    base_model = BayesianPersonalizedRanking(
        factors=50,
        learning_rate=0.01,
        regularization=0.01,
        iterations=100,
        random_state=None
    )

    base_model.fit(train_matrix)

    p_mean, p_std = [], []
    r_mean, r_std = [], []

    for k in ks:

        model_fn = lambda: base_model

        pm, ps, rm, rs = evaluate_mean_std(
            model_fn,
            train_matrix,
            test_dict,
            k,
            n_runs=n_runs,
            n_jobs=n_jobs
        )

        p_mean.append(pm)
        p_std.append(ps)
        r_mean.append(rm)
        r_std.append(rs)

        print(f"K={k} | P={pm:.4f}±{ps:.4f} | R={rm:.4f}±{rs:.4f}")

    return ks, p_mean, p_std, r_mean, r_std

# ============================================================
# RUN
# ============================================================

def run_dataset(name, df):

    print("\n====================")
    print(name)
    print("====================")

    train_df, test_df = train_test_split_userwise(df)
    full_df = pd.concat([train_df, test_df])

    matrix, user_map, item_map = create_matrix(full_df)

    train_rows = train_df.user.map(user_map)
    train_cols = train_df.item.map(item_map)

    train_matrix = sp.csr_matrix(
        (np.ones(len(train_df)), (train_rows, train_cols)),
        shape=matrix.shape
    )

    test_dict = build_test_dict(test_df, user_map, item_map)

    # ---------------- FACTORS ----------------
    f, p, p_std, r, r_std = experiment_factors(train_matrix, test_dict)

    plot_mean_std(f, p, p_std, "Factors", "Precision@10",
            f"{name} Precision@10", f"{name}_precision_factors.png")

    plot_mean_std(f, r, r_std, "Factors", "Recall@10",
            f"{name} Recall@10", f"{name}_recall_factors.png")

    # ---------------- K ----------------
    k, p, p_std, r, r_std = experiment_k(train_matrix, test_dict)

    plot_mean_std(k, p, p_std, "K", "Precision@K",
            f"{name} Precision@K", f"{name}_precision_k.png")

    plot_mean_std(k, r, r_std, "K", "Recall@K",
            f"{name} Recall@K", f"{name}_recall_k.png")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading LastFM...")
    lastfm = load_lastfm(LASTFM_PATH)

    print("Loading MovieLens...")
    movielens = load_movielens(MOVIELENS_PATH)

    run_dataset("LastFM", lastfm)
    run_dataset("MovieLens", movielens)

    print("\nDONE")