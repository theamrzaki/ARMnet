import pandas as pd

# Load CSV
csv_file = "results/RQ2_detailed.csv"
df = pd.read_csv(csv_file)

# Drop rows that are fully empty (the last blank line)
df = df.dropna(subset=["Generator", "Judge", "Overall Avg"])

df = df[~df['Generator'].isin(["_home_db2003_Desktop_Amr_Tests_RCAEval_saved_models_final_model_2",
                               "_home_db2003_Desktop_Amr_Tests_RCAEval_saved_models_final_model"])]

# Map generator names to readable form
def readable_generator(name):
    # Rename ARMnet
    if name == "_home_db2003_Desktop_Amr_Tests_RCAEval_saved_models_final_model_3":
        return "ARMnet LLM"
    # Otherwise, take first part before '_'
    return name.split('_')[0] if '_' in name else name

df['Generator'] = df['Generator'].apply(readable_generator)

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

# dont print the final position column
print("\n🏆 Majority Voting Results (without Final Position):\n")
print(rank_summary[["Generator", "avg_rank", "avg_score"]])

# save in latex table format
with open("Images/majority_voting_results.tex", "w") as f:
    f.write(rank_summary[["Generator", "avg_rank", "avg_score"]].to_latex(index=False, float_format="%.2f"))

# Save for reference
rank_summary.to_csv("Images/majority_voting_results.csv", index=False)