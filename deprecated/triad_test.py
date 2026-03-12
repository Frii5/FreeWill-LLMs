import pandas as pd
import itertools

df = pd.read_csv("data.csv")
means = df["med"].values

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

# generate all configurations
for de_perm in itertools.permutations(DE):
    for du_perm in itertools.permutations(DU):

        triads = []
        pairs = set()
        total_spread = 0

        for i in range(5):
            j = de_perm[i]
            k = du_perm[i]

            fw = means[i]
            de = means[5+j]
            du = means[10+k]

            spread = max(fw, de, du) - min(fw, de, du)
            total_spread += spread

            triads.append((i, j, k, spread))
            pairs |= get_pairs(i, j, k)

        results.append((total_spread, triads, pairs))

# sort configurations by spread
results.sort(key=lambda x: x[0])

selected_blocks = []
used_pairs = set()

for score, triads, pairs in results:

    if pairs.isdisjoint(used_pairs):
        selected_blocks.append((score, triads, pairs))
        used_pairs |= pairs

    if len(selected_blocks) == 3:
        break

# print results
for idx, (score, triads, _) in enumerate(selected_blocks, 1):
    print(f"\nBlock {idx}")
    print("Total spread:", score)
    for t in triads:
        print(t)