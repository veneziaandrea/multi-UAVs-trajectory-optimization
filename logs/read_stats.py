import pandas as pd

# Load the database
df = pd.read_csv("switching_stats.csv")

# Group the data by Algorithm, and calculate the MEAN across all maps and drones
summary = df.groupby("Algorithm").mean(numeric_only=True)

# Drop the Map_Seed column from the summary (since averaging the seed ID is meaningless)
summary = summary.drop(columns=["Map_Seed"])

print("\n=== AVERAGE PERFORMANCE ACROSS ALL MAPS ===")
print(summary.to_string())