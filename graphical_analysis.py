import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_data(df, domain):
    for column in df.columns:
        plt.figure(figsize=(7, 4))
        if df[column].dtype == 'object':
            sns.countplot(x=column, data=df)
        else:
            sns.kdeplot(df[column], fill=True)
        plt.title(f"{column} Distribution in {domain} Dataset")
        plt.tight_layout()
        plt.show()
