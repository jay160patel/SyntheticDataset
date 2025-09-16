import os
import pandas as pd


def load_or_generate_real_data(domain, synthetic_df):
    file_name = f"real_{domain.replace(' ', '_').lower()}.csv"
    if os.path.exists(file_name):
        print(f"📥 Real dataset loaded: {file_name}")
        return pd.read_csv(file_name)

    print(f"⚠ No real dataset found for {domain}. Generating dummy real dataset instead.")
    print("🔧 Using synthetic data to simulate real data for comparison.")
    return synthetic_df.copy()
