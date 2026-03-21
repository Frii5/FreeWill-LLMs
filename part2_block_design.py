import pandas as pd
import itertools

df = pd.read_csv("results_SDS/sds_scores_part2.csv")
scores = df["med"].values

FC = range(7)
MC = range(7)

results = []

def get_pair(i, j):
    return ("FC", i, "MC", j)

for mc_perm in itertools.permutations(MC):
    dyads = []
    pairs = set()
    total_spread = 0
    spreads = []

    for i in range(7):
        j = mc_perm[i]

        fc = scores[i]
        mc = scores[7 + j]

        spread = abs(fc - mc)
        total_spread += spread
        spreads.append(spread)

        dyads.append((i, j, spread))
        pairs.add(get_pair(i, j))

    uniformity = max(spreads) - min(spreads)
    results.append((total_spread, uniformity, dyads, pairs))

spread_levels = sorted(set(r[0] for r in results))
allowed_spreads = set(spread_levels[:10])

candidate_blocks = [r for r in results if r[0] in allowed_spreads]
candidate_blocks.sort(key=lambda x: (x[0], x[1]))

selected_blocks = []
used_pairs = set()

for score, uniformity, dyads, pairs in candidate_blocks:
    if pairs.isdisjoint(used_pairs):
        selected_blocks.append((score, uniformity, dyads))
        used_pairs |= pairs

    if len(selected_blocks) == 7:
        break

for idx, (score, uniformity, dyads) in enumerate(selected_blocks, 1):
    print(f"\nBlock {idx}")
    print("Total spread:", score)
    print("Uniformity:", uniformity)
    for i, j, spread in dyads:
        print((f"FC{i+1}", f"MC{j+1}", spread))
