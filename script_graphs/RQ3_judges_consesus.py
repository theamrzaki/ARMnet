import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Load your CSV data
# =========================
csv_file = "results/RQ2_detailed.csv"  # Replace with your CSV path
df = pd.read_csv(csv_file)

# =========================
# Map generator names (split by first '_') and judge names (split by '/')
# =========================
def map_generator(gen_name):
    if str(gen_name) == "nan":
        return gen_name
    if "_" in gen_name:
        return gen_name.split("_")[0]
    elif "/" in gen_name:
        return gen_name.split("/")[0]
    else:
        return gen_name

def map_judge(judge_name):
    if str(judge_name) == "nan":
        return judge_name
    if "/" in judge_name:
        return judge_name.split("/")[0]
    else:
        return judge_name

df["generator_readable"] = df["Generator"].apply(map_generator)
df["judge_readable"] = df["Judge"].apply(map_judge)

# =========================
# Metrics mapping
# =========================
metrics = {
    "feasibility": "mitigation plan can be implemented as that it is feasible",
    "efficiency": "mitigation plan is efficient as that it solves the root cause",
    "scalability": "mitigation plan is scalable as that it can be applied for large-scale issues",
    "clarity": "mitigation plan is clear for the operator as that it provides detailed steps and guidance",
    "overall_avg": "Overall Avg"
}

# =========================
# Plot heatmaps for all metrics
# =========================
for metric_key, col_name in metrics.items():
    # Pivot table with mean aggregation to handle duplicates
    heatmap_data = df.pivot_table(
        index="generator_readable",
        columns="judge_readable",
        values=col_name,
        aggfunc="mean"
    )
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={'label': col_name}
    )
    plt.title(f"Heatmap of {metric_key.capitalize()} Scores")
    plt.ylabel("Generator")
    plt.xlabel("Judge")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
