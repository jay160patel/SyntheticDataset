Synthetic Data Factory Scalable and Domain-Agnostic Data Generation with Generative AI and Statistical Fidelity

Overview The Synthetic Data Factory is a robust, scalable framework designed to generate high-quality synthetic datasets across multiple domains including Healthcare, Finance, Retail, and the Stock Market. By leveraging Generative Adversarial Networks (GANs) like CTGAN and statistical models such as Gaussian Copulas, this project aims to produce privacy-preserving and utility-rich synthetic data that maintains the statistical characteristics of real data.

🎯Features ✅ Domain-Agnostic Design: Plug-and-play architecture to support multiple domains:

1)Advanced Generative Models: Supports CTGAN, Gaussian Copulas, and other models.

2)Statistical Fidelity Checks: Uses tests like the Kolmogorov–Smirnov test, correlation comparison, and distribution analysis.

3)Data Utility Evaluation Module: Compares real and synthetic data by training ML models and evaluating metrics like accuracy, F1-score, and AUC-ROC.

4)Privacy-Aware Architecture: Optionally integrates Differential Privacy for secure synthetic data.

5)Multi-source Ingestion: Accepts CSVs, databases, APIs, and external data portals (e.g., Kaggle, UCI).

6)Real-time Streaming (Future Scope): Planned module for dynamic, real-time synthetic data generation.

🧬 Use Cases:

1)Medical research using synthetic patient records.

2)Financial simulations without exposing sensitive data.

3)Retail demand forecasting with anonymized customer behaviors.

4)Time-series model testing for trading algorithms.

🛠️ Technologies Used Python 3.10+

1)CTGAN / SDV

2)Gaussian Copulas

3)Scikit-learn, XGBoost

4)Pandas, NumPy, Matplotlib, Seaborn

5)Kolmogorov–Smirnov Test, Jensen-Shannon Divergence

6)Streamlit (for visualization, optional)

7)FastAPI (optional backend support)

📈 Evaluation Metrics Distribution Similarity (KS-Test, Wasserstein Distance)

1)Correlation Matrix Heatmaps

2)Classifier Performance (Accuracy, Precision, Recall, F1-score, AUC)

3)ROC Curve visualizations
