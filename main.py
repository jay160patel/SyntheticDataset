import pandas as pd
from synthetic_data_generator import generate_synthetic_dataset
from evaluation_metrics import evaluate_models
from statistical_fidelity import compute_js_divergence
from real_data_loader import load_or_generate_real_data
from graphical_analysis import visualize_data


def main():
    print("\n📊 Synthetic Data Factory: Scalable and Domain-Agnostic Data Generation")
    print("📂 Available Domains: Healthcare, Finance, Retail, Stock Market")

    domain = input("Select a domain: ").strip()
    num_samples = input(f"Enter number of synthetic samples for {domain}: ")

    try:
        num_samples = int(num_samples)
    except ValueError:
        print("❌ Invalid input! Number of samples must be an integer.")
        return

    print("\n🔄 Generating synthetic dataset...")
    synthetic_df = generate_synthetic_dataset(domain, num_samples)
    print("✅ Synthetic dataset created.")

    print("\n📝 Sample synthetic data:")
    print(synthetic_df.head())

    print("\n📈 Visualizing synthetic data...")
    visualize_data(synthetic_df, domain)

    real_df = load_or_generate_real_data(domain, synthetic_df)

    print("\n📊 Statistical Fidelity Analysis:")
    for col in synthetic_df.columns:
        if pd.api.types.is_numeric_dtype(synthetic_df[col]):
            js = compute_js_divergence(real_df[col], synthetic_df[col])
            print(f"  ➤ JS Divergence ({col}): {js:.4f}")

    print("\n🤖 Evaluating Data Utility via ML Models:\n")
    evaluate_models(real_df, synthetic_df, domain)


if __name__ == "__main__":
    main()
