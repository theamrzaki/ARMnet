import pandas as pd

# Load CSV
df = pd.read_csv("evaluation_results.csv")

# Drop rows that are fully empty (the last blank line)
df = df.dropna(subset=["Generator", "Judge", "Overall Avg"])

# Convert numeric columns
df["Overall Avg"] = pd.to_numeric(df["Overall Avg"], errors="coerce")

# Rank per Judge (1 = best)
df["Rank"] = df.groupby("Judge")["Overall Avg"].rank(ascending=False, method="average")

# Compute the average rank per Generator across all judges
rank_summary = (
    df.groupby("Generator")
      .agg(avg_rank=("Rank", "mean"), avg_score=("Overall Avg", "mean"))
      .sort_values("avg_rank")
      .reset_index()
)

# Add a majority-vote order column
rank_summary["Final Position"] = rank_summary["avg_rank"].rank(method="dense").astype(int)

print("\n🏆 Majority Voting Results (lower rank = better):\n")
print(rank_summary[["Final Position", "Generator", "avg_rank", "avg_score"]])

# Save for reference
rank_summary.to_csv("majority_voting_results.csv", index=False)