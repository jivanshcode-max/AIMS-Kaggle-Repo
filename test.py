import pandas as pd

# Load your large dataset
df = pd.read_csv("melb_data.csv")

# Random sample of 2000 rows
sample_df = df.sample(n=2000, random_state=1)

# Save to new CSV
sample_df.to_csv("melb_data_S.csv", index=False)
