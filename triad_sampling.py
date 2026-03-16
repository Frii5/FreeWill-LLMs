import pandas as pd
import itertools

df = pd.read_csv("data.csv")
means = df["med"].values   # or "avg"

FW = range(5)
DE = range(5)
DU = range(5)

results = []

def get_pairs(i, j, k):
    return {
        ("FW", i, "DE", j),
        ("FW", i, "DU", k),
        ("DE", j, "DU", k)
    }

for de_perm in itertools.permutations(DE):
    for du_perm in itertools.permutations(DU):
        triads = []
        pairs = set()
        total_spread = 0
        spreads = []

        for i in range(5):
            j = de_perm[i]
            k = du_perm[i]

            fw = means[i]
            de = means[5 + j]
            du = means[10 + k]

            spread = max(fw, de, du) - min(fw, de, du)
            total_spread += spread
            spreads.append(spread)

            triads.append((i, j, k, spread))
            pairs |= get_pairs(i, j, k)

        uniformity = max(spreads) - min(spreads)
        
        results.append((total_spread, uniformity, triads, pairs))

# keep only best-spread blocks
best_score = min(r[0] for r in results)
best_blocks = [r for r in results if r[0] == best_score]

# among tied best blocks, prefer most uniform ones
best_blocks.sort(key=lambda x: x[1])

selected_blocks = []
used_pairs = set()

for score, uniformity, triads, pairs in best_blocks:
    if pairs.isdisjoint(used_pairs):
        selected_blocks.append((score, uniformity, triads))
        used_pairs |= pairs

    if len(selected_blocks) == 4:
        break
    
for idx, (score, uniformity, triads) in enumerate(selected_blocks, 1):
    print(f"\nBlock {idx}")
    print("Total spread:", score)
    print("Uniformity:", uniformity)
    for t in triads:
        print(t)
