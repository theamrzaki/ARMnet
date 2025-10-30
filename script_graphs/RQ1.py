import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# -----------------------------
# Load data
# -----------------------------
csv = "results/RQ1.csv"
df = pd.read_csv(csv)
metrics = ["cpu", "mem",  "socket", "delay", "loss", "disk"]
# if dataset has "re1", delete the socket column
df.loc[df["dataset"].str.contains("re1"), "socket"] = None
agg_df = df.dropna(axis=1, how='all')

# Aggregate by mean
# drop baro with baro = true
df = df[~((df["method"] == "baro") & (df["with_baro_post"] == True))]
# only if method is "RMDnet" combine method column with model_class column
df.loc[df["method"] == "RMDnet", "method"] = df["model_class"]
agg_df = df.groupby(["method", "dataset", "with_baro_post"])[metrics].mean().reset_index()


# Make with_baro readable
agg_df["with_baro_post"] = agg_df["with_baro_post"].astype(str)
# if with baro (make it "w" if true else "w/o") (except for case of method = "baro", then keep as is)
agg_df.loc[agg_df["with_baro_post"] == "True", "with_baro_post"] = "w"
agg_df.loc[agg_df["with_baro_post"] == "False", "with_baro_post"] = "w/o"
agg_df.loc[agg_df["method"] == "baro", "with_baro_post"] = ""
agg_df["with_baro_post"] = agg_df["with_baro_post"].replace({"True": "w", "False": "w/o","":""})
# for other methods, make method_combo as "method (with_baro_post)"
agg_df["method_combo"] = agg_df["method"] + " (" + agg_df["with_baro_post"]+")"

#if method combo has "baro (", then just make it "baro"
agg_df.loc[agg_df["method_combo"].str.contains("baro \("), "method_combo"] = "baro"

# order methods ["baro", "iTransformer", "TimeMixerpp","Dlinear","fits_linear"] and with_baro = ["w/o", "w"]
method_order = ["baro", "Dlinear","Fits","iTransformer", "TimeMixerpp"]
with_baro_order = ["w/o", "w"]
agg_df["method"] = pd.Categorical(agg_df["method"], categories=method_order, ordered=True)
agg_df["with_baro_post"] = pd.Categorical(agg_df["with_baro_post"], categories=with_baro_order, ordered=True)
agg_df = agg_df.sort_values(by=["dataset", "method", "with_baro_post"])




# Prepare figure layout
# -----------------------------
datasets = agg_df["dataset"].unique()
n = len(datasets)
cols = 2  # number of columns of subplots
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
axes = axes.flatten()

# -----------------------------
# Plot each dataset
# -----------------------------
# order datasets so that they appear as
"""
Re1 | RE2
RE1 | RE2
RE1 | RE2
"""
categories = {
    "Statistical": ["baro"],
    "Linear": ["Dlinear (w)", "Dlinear (w/o)", "Fits (w)", "Fits (w/o)"],
    "Transformer": ["iTransformer (w)", "iTransformer (w/o)", "TimeMixerpp (w)", "TimeMixerpp (w/o)"]
}
desired_order = ["re1-ob","re2-ob", "re1-tt","re2-tt","re1-ss",  "re2-ss"]  # put the datasets in the order you want
datasets = [d for d in desired_order if d in agg_df["dataset"].unique()]
for i, dataset in enumerate(datasets):
    dataset_df = agg_df[agg_df["dataset"] == dataset]

    heatmap_data = dataset_df.set_index("method_combo")[metrics]

    
    sns.heatmap(
        heatmap_data,
        annot=heatmap_data, fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Mean Value"},
        ax=axes[i]
    )


    axes[i].set_title(f"{dataset.upper()}", fontsize=14, weight="bold")
    axes[i].set_xlabel("Metric")
    #axes[i].set_ylabel("Method | BARO Setting")
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].tick_params(axis='y', rotation=0)
    #save each subplot as a separate pdf under formde ouptut/figures/RQ1_<dataset>.pdf
    # make the fig on its own
    #if dataset has re1, remove socket column
    if "re1" in dataset:
        heatmap_data = heatmap_data.drop(columns=["socket"])
    fig = plt.figure()
    ax = sns.heatmap(
        heatmap_data,
        annot=heatmap_data, fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Mean Value"},
    )

    # Outer category labels
    #ordered_rows = sum(categories.values(), [])
    #ypos = 0
    #for cat, models in categories.items():
    #    start = ypos
    #    end = ypos + len(models) - 1
    #    y_center = (start + end)/2
    #    ax.text(-0.3,         1 - y_center/len(ordered_rows) - 0.1, cat, rotation=90, va='center', ha='center', fontsize=5, transform=ax.transAxes)
    #    ypos += len(models)

    
    # -----------------------------
    # Draw horizontal lines outside the heatmap to separate categories
    # -----------------------------
    category_boundaries = []
    counter = 0
    for cat, models in categories.items():
        counter += len(models)
        category_boundaries.append(counter)

    # Convert heatmap row indices to figure coordinates
    for boundary in category_boundaries[:-1]:
        ax.plot(
            [-2, heatmap_data.shape[1]],  # full width of heatmap
            [boundary, boundary],                # row index boundary
            color='black',
            linestyle='--',
            linewidth=1,
            clip_on=False                        # ensures line draws outside cells
        )


    plt.title(f"{dataset.upper()}", fontsize=14, weight="bold")
    plt.xlabel("Metric")
    #plt.ylabel("Method | BARO Setting")
    # remove y label
    plt.ylabel("")
    plt.tick_params(axis='x', rotation=30)
    plt.tick_params(axis='y', rotation=0)
    plt.savefig(f"Images/RQ1_{dataset}.pdf", bbox_inches='tight')

    # print the contents of each subplot and save it to a text file
    with open(f"Images/RQ1_{dataset}.txt", "w") as f:
        f.write(f"Dataset: {dataset.upper()}\n")
        f.write(heatmap_data.to_string())   
        

plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# Compute average performance improvement (w vs w/o) per dataset
# -----------------------------------------------------------
improvement_summary = []
# remove datasets with tt
agg_df = agg_df[~agg_df["dataset"].str.contains("tt")]
# Iterate over datasets
for dataset in agg_df["dataset"].unique():
    df_dataset = agg_df[agg_df["dataset"] == dataset]

    # For each method in this dataset
    for method in df_dataset["method"].unique():
        df_method = df_dataset[df_dataset["method"] == method]

        # Only compute if both 'w' and 'w/o' exist
        if "w" in df_method["with_baro_post"].values and "w/o" in df_method["with_baro_post"].values:
            df_w = df_method[df_method["with_baro_post"] == "w"][metrics].mean().mean()
            df_wo = df_method[df_method["with_baro_post"] == "w/o"][metrics].mean().mean()
            improvement = ((df_w - df_wo) / df_wo) * 100 if df_wo != 0 else 0

            improvement_summary.append({
                "dataset": dataset,
                "method": method,
                "avg_w": df_w,
                "avg_wo": df_wo,
                "improvement_%": improvement
            })

# Create DataFrame
improvement_df = pd.DataFrame(improvement_summary)

# Print summary
print("\n=== Average Improvement of 'with BARO' over 'without BARO' (per dataset) ===")
print(improvement_df.to_string(index=False))

# Save results
improvement_df.to_csv("Images/RQ1_baro_improvement_per_dataset.csv", index=False)

# -----------------------------------------------------------
# (Optional) Visualization: improvement per dataset
# -----------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.barplot(
    data=improvement_df,
    x="dataset",
    y="improvement_%",
    hue="method"
)
plt.title("Average Improvement (%) of 'with BARO' over 'without BARO' per Dataset")
plt.ylabel("Improvement (%)")
plt.xlabel("Dataset")
plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("Images/RQ1_baro_improvement_per_dataset.pdf", bbox_inches="tight")
plt.show()
