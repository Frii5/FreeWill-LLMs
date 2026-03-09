import pandas as pd
import itertools

# load data
df = pd.read_csv("data.csv")

# assume column "avg" contains desirability means
means = df["avg"].values

# split constructs
FW = means[0:5]
DE = means[5:10]
DU = means[10:15]

best_score = float("inf")
best_triads = None

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

        if total_spread < best_score:
            best_score = total_spread
            best_triads = triads

print("Best total spread:", best_score)
print("Triads (FW, DE, DU):")

for t in best_triads:
    print(t)