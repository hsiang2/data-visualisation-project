import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
all_df=pd.read_csv("mc1-reports-data.csv")
all_df.head()

all_df["time"] = pd.to_datetime(all_df["time"])
all_df["time"].min(), all_df["time"].max()

all_df["location"].value_counts().sort_index()

all_df.set_index("time").resample("6H").mean()["power"].plot(figsize=(10,4), title="Power Problem Over Time")

grouped = all_df.groupby("location")

report_count = grouped.size()

missing_ratio = grouped.apply(lambda g: g.isnull().mean().mean())

reliability_score = report_count / (1 + missing_ratio * 10)

rel_df = pd.DataFrame({
    "location": report_count.index,
    "report_count": report_count.values,
    "missing_ratio": missing_ratio.values,
    "reliability_score": reliability_score.values
}).reset_index(drop=True)

plt.figure(figsize=(12,6))
plt.bar(rel_df["location"].astype(str), rel_df["reliability_score"], color='skyblue')
plt.title("Reliability scores of community reports")
plt.xlabel("Location")
plt.ylabel("Reliability Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def corr_shake_buildings(group):
    if group["shake_intensity"].notnull().sum() < 2 or group["buildings"].notnull().sum() < 2:
        return np.nan
    return group["shake_intensity"].corr(group["buildings"])

corr_values = grouped.apply(corr_shake_buildings)

rel_df["corr_shake_buildings"] = rel_df["location"].map(corr_values)

print(rel_df.sort_values("corr_shake_buildings", ascending=False).head(10))

indicators = ["sewer_and_water", "power", "roads_and_bridges", "medical", "buildings", "shake_intensity"]

results = []

for indicator in indicators:
    grouped = all_df.groupby("location")
    total_count = grouped.size()
    non_missing_count = grouped[indicator].apply(lambda x: x.notnull().sum())
    completeness = non_missing_count / total_count

    df_indicator = pd.DataFrame({
        "location": total_count.index,
        "total_count": total_count.values,
        "non_missing_count": non_missing_count.values,
        "completeness": completeness.values,
    })
    df_indicator["indicator"] = indicator
    results.append(df_indicator)

reliability_df = pd.concat(results, ignore_index=True)

pivot_df = reliability_df.pivot(index="location", columns="indicator", values="completeness")
print(pivot_df)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_df, annot=True, cmap="viridis", fmt=".3f")
plt.title("Data completeness rate (reliability) of each indicator in each community")
plt.xlabel("Data")
plt.ylabel("Community (Location)")
plt.tight_layout()
plt.show()

indicators = ["sewer_and_water", "power", "roads_and_bridges", "medical", "buildings", "shake_intensity"]

start_date = "2020-04-06"
end_date = "2020-04-09 23:59:59"
mask = (all_df["time"] >= start_date) & (all_df["time"] <= end_date)
df_filtered = all_df.loc[mask].copy()

df_filtered.set_index("time", inplace=True)
resample_rule = "1H"

def missing_ratio_for_period(sub_df):
    total_count = len(sub_df)
    if total_count == 0:
        return pd.Series({col: np.nan for col in indicators})
    ratio_dict = {}
    for col in indicators:
        missing_count = sub_df[col].isnull().sum()
        ratio_dict[col] = missing_count / total_count
    return pd.Series(ratio_dict)

missing_by_time = df_filtered.resample(resample_rule).apply(missing_ratio_for_period)

plt.figure(figsize=(12, 6))

for col in indicators:
    plt.plot(missing_by_time.index, missing_by_time[col], label=col)

plt.title("Changes in missing rates of each indicator over time")
plt.xlabel("Time")
plt.ylabel("Missing rates (0~1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

all_df.isnull().mean()

all_df["medical_reported"] = all_df["medical"].notnull().astype(int)

all_df["shake_intensity"] = all_df.groupby("location")["shake_intensity"].transform(
    lambda x: x.fillna(x.median())
)

all_df["sewer_and_water"] = all_df["sewer_and_water"].fillna(all_df["sewer_and_water"].median())
all_df["buildings"] = all_df["buildings"].fillna(all_df["buildings"].median())

plt.figure(figsize=(12, 4))
report_counts = all_df.set_index("time").resample("5T").size()
plt.plot(report_counts.index, report_counts.values, marker='o', linestyle='-', color='blue')
plt.title("Report count over time (5-minute interval)")
plt.xlabel("Time")
plt.ylabel("Number of reports")
plt.grid(True)
plt.show()

location_power = all_df.groupby("location")["power"].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(location_power.index.astype(str), location_power.values, color='salmon')
plt.title("Average severity of electricity problems in each community")
plt.xlabel("Community (Location)")
plt.ylabel("Average power problem")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

top5_locations = all_df.groupby("location")["shake_intensity"].mean().sort_values(ascending=False).head(5).index
data_to_plot = [all_df[all_df["location"] == loc]["shake_intensity"] for loc in top5_locations]
plt.figure(figsize=(10, 6))
plt.boxplot(data_to_plot, labels=[str(loc) for loc in top5_locations])
plt.title("Distribution of earthquake intensity (Top 5 communities with the highest earthquake intensity)")
plt.xlabel("Communities (Location)")
plt.ylabel("Earthquake intensity")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist2d(all_df["shake_intensity"], all_df["power"], bins=11, range=[[0,10],[0,10]], cmap="plasma")
plt.colorbar(label="Count")
plt.title("Severity of power issues vs. intensity of earthquakes felt (2D histogram)")
plt.xlabel("Intensity of earthquakes")
plt.ylabel("Severity of power issues")
plt.show()

damage_vars = ["shake_intensity", "sewer_and_water", "power", "roads_and_bridges", "medical", "buildings"]
corr_matrix = all_df[damage_vars].corr()
plt.figure(figsize=(8, 6))
im = plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(damage_vars)), damage_vars, rotation=45)
plt.yticks(range(len(damage_vars)), damage_vars)
plt.title("Heat map of correlation between various disaster indicators")
plt.tight_layout()
plt.show()

all_df["time"] = pd.to_datetime(all_df["time"])

start_date = "2020-04-06"
end_date = "2020-04-08 23:59:59"
mask = (all_df["time"] >= start_date) & (all_df["time"] <= end_date)
filtered_df = all_df[mask].copy()

filtered_df.set_index("time", inplace=True)

report_counts = filtered_df.resample("1H").size()

avg_power = filtered_df.resample("1H")["power"].mean()

avg_buildings = filtered_df.resample("1H")["buildings"].mean()

plt.figure(figsize=(12, 4))
plt.plot(report_counts.index, report_counts.values, marker='o', linestyle='-')
plt.title("【4/6 - 4/8】The number of disaster reports changes over time")
plt.xlabel("Time")
plt.ylabel("Number of reports")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(avg_power.index, avg_power.values, marker='o', label="Power issue (power)", color='red')
plt.plot(avg_buildings.index, avg_buildings.values, marker='s', label="Building damage (buildings)", color='blue')

plt.title("【4/6 - 4/8】Power issues & building damage severity varies over time")
plt.xlabel("Time")
plt.ylabel("Damage severity (0-10)")
plt.legend()
plt.grid(True)
plt.show()

start_date = "2020-04-06"
end_date = "2020-04-09 23:59:59"
mask = (all_df["time"] >= start_date) & (all_df["time"] <= end_date)
df_event = all_df.loc[mask].copy()

df_event.set_index("time", inplace=True)
resampled = df_event.resample("1H").agg({
    "power": "mean",
    "buildings": "mean",
    "shake_intensity": "mean",
    "roads_and_bridges": "mean",
    "sewer_and_water": "mean"
})
report_counts = df_event.resample("1H").size()

plt.figure(figsize=(14, 8))

plt.plot(resampled.index, resampled["power"], label="Avg Power Damage", color="red")
plt.plot(resampled.index, resampled["buildings"], label="Avg Building Damage", color="blue")
plt.plot(resampled.index, resampled["shake_intensity"], label="Avg Shake Intensity", color="green")
plt.plot(resampled.index, resampled["roads_and_bridges"], label="Avg Roads/Bridges Damage", color="orange")
plt.plot(resampled.index, resampled["sewer_and_water"], label="Avg Sewer/Water Damage", color="purple")

normalized_counts = report_counts / report_counts.max() * 10
plt.plot(report_counts.index, normalized_counts, label="Normalized Report Count", color="black", linestyle="--")

main_shock = pd.to_datetime("2020-04-08 03:00:00")
aftershock = pd.to_datetime("2020-04-09 06:00:00")
plt.axvline(x=main_shock, color="grey", linestyle="--", linewidth=2, label="Main Shock")
plt.axvline(x=aftershock, color="brown", linestyle="--", linewidth=2, label="Aftershock")

plt.title("Global Event Progression: Comprehensive Multi-Indicator Analysis")
plt.xlabel("Time")
plt.ylabel("Average Damage / Normalized Report Count")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

damage_cols = ["sewer_and_water", "power", "roads_and_bridges", "medical", "buildings", "shake_intensity"]

grouped = all_df.groupby("location")[damage_cols].mean()

grouped["avg_severity"] = grouped.mean(axis=1)

sorted_df = grouped.sort_values("avg_severity", ascending=False)
print("Ranking of the most severely affected communities:")
print(sorted_df.head(10))

plt.figure(figsize=(12, 6))
plt.bar(sorted_df.index.astype(str), sorted_df["avg_severity"], color="salmon")
plt.title("Comparison of comprehensive disaster situations in communities")
plt.xlabel("Location")
plt.ylabel("Average damage rating (avg_severity)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

grouped["avg_severity"] = grouped.mean(axis=1)

weights = {
    "power": 2.0,
    "buildings": 2.0,
    "roads_and_bridges": 1.5,
    "medical": 1.5,
    "sewer_and_water": 1.0,
    "shake_intensity": 1.0
}

total_weight = sum(weights.values())
grouped["weighted_score"] = (
    grouped["power"] * weights["power"] +
    grouped["buildings"] * weights["buildings"] +
    grouped["roads_and_bridges"] * weights["roads_and_bridges"] +
    grouped["medical"] * weights["medical"] +
    grouped["sewer_and_water"] * weights["sewer_and_water"] +
    grouped["shake_intensity"] * weights["shake_intensity"]
) / total_weight

sorted_weighted = grouped.sort_values("weighted_score", ascending=False)
print("Top 10 communities ranked by weighted composite score:")
print(sorted_weighted.head(10)[["weighted_score", "avg_severity"]])

plt.figure(figsize=(12, 6))
plt.bar(sorted_weighted.index.astype(str), sorted_weighted["weighted_score"], color="salmon")
plt.title("Weighted comprehensive disaster score for each community")
plt.xlabel("Community (Location)")
plt.ylabel("Weighted Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

nuclear_df = all_df[all_df["location"] == 4].copy()

report_count = nuclear_df.shape[0]
mean_power = nuclear_df["power"].mean()
mean_buildings = nuclear_df["buildings"].mean()
mean_shake = nuclear_df["shake_intensity"].mean()

print("Nuclear power plant area (Location 4) data analysis:")
print("Total reports:", report_count)
print("Average power problem severity:", round(mean_power, 2))
print("Average building damage severity:", round(mean_buildings, 2))
print("Average earthquake intensity:", round(mean_shake, 2))

start_date = "2020-04-06"
end_date = "2020-04-09 23:59:59"
mask = (nuclear_df["time"] >= start_date) & (nuclear_df["time"] <= end_date)
nuclear_filtered = nuclear_df.loc[mask].copy()

nuclear_filtered.set_index("time", inplace=True)
power_ts = nuclear_filtered.resample("1H")["power"].mean()
shake_ts = nuclear_filtered.resample("1H")["shake_intensity"].mean()

plt.figure(figsize=(12, 6))
plt.plot(power_ts.index, power_ts.values, marker='o', linestyle='-', color="red", label="Average power problem")
plt.plot(shake_ts.index, shake_ts.values, marker='s', linestyle='-', color="green", label="Average earthquake intensity")
plt.title("Changes of key indicators of nuclear power plant area (Location 4) over time")
plt.xlabel("Time")
plt.ylabel("Average severity (0-10)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()