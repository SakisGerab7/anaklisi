import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')

# ------------------------------------
# Read Gutenberg file
# ------------------------------------

with open("pg1342.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ------------------------------------
# Keep only novel text
# ------------------------------------

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***"

start = text.find(start_marker)
end = text.find(end_marker)

text = text[start+len(start_marker):end-len(end_marker)].lower()

text = text.replace("_", "")
text = text.replace("’", "'")

# ------------------------------------
# Tokenization (NO STEMMING)
# ------------------------------------

words = nltk.tokenize.word_tokenize(text)

words = [w for w in words if re.match(r"^[a-zA-Z']+$", w)]

# ------------------------------------
# Frequencies
# ------------------------------------

freq = Counter(words)

print("Total unique terms =", len(freq))

sorted_terms = sorted(
    freq.items(),
    key=lambda x: x[1],
    reverse=True
)

ranks = np.array(range(1, len(sorted_terms) + 1))
freqs = np.array([x[1] for x in sorted_terms])

# -------------------------------------
# Estimate Zipf constant by calculating mean of rank * frequency for top 1000 terms
# -------------------------------------
constants = ranks * freqs
zipf_const_mean = np.mean(constants)

print(f"\nMean of Rank * Frequency for top 1000 terms = {zipf_const_mean:.2f}")

# ------------------------------------
# Estimate Zipf constant and exponent by fitting a line to log-log data
# ------------------------------------
log_ranks = np.log(ranks)
log_freqs = np.log(freqs)

slope, intercept = np.polyfit(log_ranks, log_freqs, 1)

zipf_const = np.exp(intercept)
zipf_exponent = -slope

print(f"\nEstimated Zipf constant (c) = {zipf_const:.2f}")
print(f"Estimated Zipf exponent (a) = {zipf_exponent:.2f}")

# ------------------------------------
# Estimate Zipf constant and exponent using logarithmic bins
# ------------------------------------
bin_edges = np.logspace(0, np.log10(len(ranks)), num=20)
bin_indices = np.digitize(ranks, bin_edges)

binned_ranks = []
binned_freqs = []

for i in range(1, len(bin_edges)):
    bin_mask = (bin_indices == i)
    if np.any(bin_mask):
        binned_ranks.append(np.mean(ranks[bin_mask]))
        binned_freqs.append(np.mean(freqs[bin_mask]))

binned_log_ranks = np.log(binned_ranks)
binned_log_freqs = np.log(binned_freqs)

binned_slope, binned_intercept = np.polyfit(binned_log_ranks, binned_log_freqs, 1)
binned_zipf_const = np.exp(binned_intercept)
binned_zipf_exponent = -binned_slope

print(f"\nEstimated Zipf constant with logarithmic bins (c) = {binned_zipf_const:.2f}")
print(f"Estimated Zipf exponent with logarithmic bins (a) = {binned_zipf_exponent:.2f}")

# ------------------------------------
# Top 50 terms
# ------------------------------------

top50 = sorted_terms[:50]

terms = [x[0] for x in top50]
counts = [x[1] for x in top50]

print("\nTop 50 terms:\n")

for rank, (term, count) in enumerate(top50, start=1):
    print(f"{rank:2d}. {term:15s} {count}")

# ------------------------------------
# Term vs Frequency graph
# ------------------------------------

plt.figure(figsize=(14,6))
plt.bar(range(50), counts)

plt.xticks(
    range(50),
    terms,
    rotation=90
)

plt.xlabel("Term")
plt.ylabel("Frequency")
plt.title("Top 50 Terms in Pride and Prejudice")
plt.tight_layout()
plt.savefig("top50_terms.png")

# ------------------------------------
# Zipf log-log plot
# ------------------------------------
zipf_freqs_mean = zipf_const_mean / ranks
zipf_freqs_fit = zipf_const * ranks ** (-zipf_exponent)
zipf_freqs_bins = binned_zipf_const * binned_ranks ** (-binned_zipf_exponent)

plt.figure(figsize=(8,6))
plt.loglog(ranks, freqs, label="Actual Frequencies", marker='o', markersize=2, linestyle='None')
plt.loglog(ranks, zipf_freqs_mean, label=f"Mean Zipf Constant Fit (c={zipf_const_mean:.2f})", linestyle='--')
plt.loglog(ranks, zipf_freqs_fit, label=f"Line Fit (c={zipf_const:.2f}, a={zipf_exponent:.2f})", linestyle='--')
plt.loglog(binned_ranks, zipf_freqs_bins, label=f"Log Bin Fit (c={binned_zipf_const:.2f}, a={binned_zipf_exponent:.2f})", linestyle='--', marker='x', markersize=5)
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Zipf Plot")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("zipf_plot.png")
