import pandas as pd
import numpy as np

# Load the dataset
file_path = "Sales_Fact (1).csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Set seed for reproducibility
np.random.seed(42)

# Apply random scaling factors to introduce significant variations
df["Booking"] = df["Booking"] * np.random.uniform(0.8, 2.5, len(df))
df["Retail"] = df["Retail"] * np.random.uniform(0.5, 3.8, len(df))
df["Billing"] = df["Billing"] * np.random.uniform(0.7, 2.6, len(df))
df["TestDrive"] = df["TestDrive"] * np.random.uniform(0.6, 2.4, len(df))

# Convert to integers to maintain real-world values
df["Booking"] = df["Booking"].astype(int)
df["Retail"] = df["Retail"].astype(int)
df["Billing"] = df["Billing"].astype(int)
df["TestDrive"] = df["TestDrive"].astype(int)

# Save the modified dataset
modified_file_path = "Modified_Sales_Fact.csv"
df.to_csv(modified_file_path, index=False)

print(f"Modified dataset saved as {modified_file_path}")
