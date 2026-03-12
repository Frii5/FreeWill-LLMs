import pandas as pd
import itertools

# load data
df = pd.read_csv("data.csv")

# assume column "med" contains desirability values
means = df["med"].values

# split constructs
FW = means[0:5]
DE = means[5:10]
DU = means[10:15]

results = []

# try all permutations
for de_perm in itertools.permutations(range(5)):
    for du_perm in itertools.permutations(range(5)):

        total_spread = 0
        triads = []

        for i in range(5):
            fw = FW[i]
            de = DE[de_perm[i]]
            du = DU[du_perm[i]]

            spread = max(fw, de, du) - min(fw, de, du)
            total_spread += spread

            triads.append((i, de_perm[i], du_perm[i], spread))

        results.append((total_spread, triads))

# sort by total spread
results.sort(key=lambda x: x[0])

# take top 5 configurations
top5 = results[:5]

for idx, (score, triads) in enumerate(top5, 1):
    print(f"\nConfiguration {idx}")
    print("Total spread:", score)
    print("Triads (FW, DE, DU):")
    for t in triads:
        print(t)