import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
csv = "results/RQ1.csv"
df = pd.read_csv(csv)

# Create output folder
os.makedirs("figures", exist_ok=True)

# Extract relevant cases
baro_df = df[df["method"] == "baro"]
rmdnet_df = df[df["method"] == "RMDnet"]

# Separate with and without Baro
rmdnet_with = rmdnet_df[rmdnet_df["with_baro_post"] == True]
rmdnet_without = rmdnet_df[rmdnet_df["with_baro_post"] == False]

# Modern color palette (Tableau colors)
colors = {
    "Baro only": "#4E79A7",         # Blue
    "ARMnet (No Baro)": "#F22B2B",  # Orange
    "ARMnet (With Baro)": "#59A14F" # Green
}

# Loop through datasets
datasets = sorted(df["dataset"].dropna().unique())
width = 0.3

for dataset in datasets:
    # Compute Baro speed (single mean per dataset)
    baro_speed = baro_df[baro_df["dataset"] == dataset]["speed"].mean()

    # Get all model classes for this dataset
    models = sorted(rmdnet_df[rmdnet_df["dataset"] == dataset]["model_class"].unique())

    # X positions (1 extra for baro)
    x = np.arange(len(models) + 1)

    # Prepare bar values
    baro_vals = [baro_speed] + [np.nan] * len(models)
    without_vals = [np.nan] + [
        rmdnet_without[(rmdnet_without["dataset"] == dataset) &
                       (rmdnet_without["model_class"] == m)]["speed"].mean()
        for m in models
    ]
    with_vals = [np.nan] + [
        rmdnet_with[(rmdnet_with["dataset"] == dataset) &
                    (rmdnet_with["model_class"] == m)]["speed"].mean()
        for m in models
    ]

    # Labels
    labels = ["Baro"] + models

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(x, baro_vals, width=width, color=colors["Baro only"], label="Baro only")
    plt.bar(x - width/2, without_vals, width=width, color=colors["ARMnet (No Baro)"], label="ARMnet (No Baro)")
    plt.bar(x + width/2, with_vals, width=width, color=colors["ARMnet (With Baro)"],
            hatch="//", edgecolor="black", label="ARMnet (With Baro)")

    # Styling
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Speed", fontsize=11)
    plt.title(f"Speed Comparison for Dataset: {dataset}", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()

    # Save plot
    save_path = f"Images/{dataset}_speed.pdf"
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved: {save_path}")

    # ======================
    # Generate LaTeX tables
    # ======================
    latex_path = "Images/results_tables.tex"
    with open(latex_path, "w") as f:
        f.write("% Auto-generated LaTeX tables for RQ1 results\n\n")

    for dataset in datasets:
        baro_speed = baro_df[baro_df["dataset"] == dataset]["speed"].mean()
        models = sorted(rmdnet_df[rmdnet_df["dataset"] == dataset]["model_class"].unique())

        # Gather values
        without_vals = [
            rmdnet_without[(rmdnet_without["dataset"] == dataset) &
                        (rmdnet_without["model_class"] == m)]["speed"].mean()
            for m in models
        ]
        with_vals = [
            rmdnet_with[(rmdnet_with["dataset"] == dataset) &
                        (rmdnet_with["model_class"] == m)]["speed"].mean()
            for m in models
        ]

        # Write LaTeX
        table = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{Speed comparison for \\texttt{{{dataset}}}}}",
            "\\begin{tabular}{l c}",
            "\\toprule",
            "Scheme & Mean Speed \\\\",
            "\\midrule",
            f"Baro only & {baro_speed:.3f} \\\\"
        ]

        for model, wout, w in zip(models, without_vals, with_vals):
            table.append(f"ARMnet ({model}, No Baro) & {wout:.3f} \\\\")
            table.append(f"ARMnet ({model}, With Baro) & {w:.3f} \\\\")

        table += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "\n"]

        with open(latex_path, "a") as f:
            f.write("\n".join(table))

    print(f"✅ All LaTeX tables saved to {latex_path}")