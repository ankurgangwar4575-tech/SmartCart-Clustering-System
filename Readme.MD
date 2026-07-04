# 🛒 SmartCart Clustering System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Agglomerative%20Clustering-F7931E)](#how-it-works)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

SmartCart Clustering System is a Streamlit app for customer segmentation. It can take a customer CSV, auto-match or fill missing fields, or let a user enter a single customer manually. The app applies the same preprocessing used during training and assigns each record to a behavioral cluster. The deployed prediction flow uses agglomerative clustering because it gives the stronger segmentation result in this project, while K-Means is kept as a baseline comparison.

## ✨ What the app does

When a user opens the website, they can:

1. Upload a customer CSV.
2. See customer-level cluster labels.
3. Review cluster counts and a quick cluster profile.
4. Download the clustered output as a CSV.

This makes the project useful for marketing analysts, retail managers, and data teams who want a simple segmentation tool instead of working directly in notebooks.

## 🚀 Main features

- 📤 File upload support for customer CSVs.
- 🧩 Auto-matching of common column names and safe default filling for missing fields.
- ✍️ Manual single-customer entry for users without a ready CSV.
- 📄 Template CSV download for easy data preparation.
- 🧠 Automatic preprocessing and feature engineering.
- 🧩 Agglomerative cluster prediction using saved model artifacts.
- 🗂️ Compact prediction table plus a full-data tab.
- 📊 Cluster distribution chart.
- 📌 Quick summary table for each cluster.
- ⬇️ Download button for the scored dataset.
- ⚡ Cached model loading for faster app startup.

## 🔄 How it works

The trained pipeline in [notebooks/SmartCartClustering.ipynb](notebooks/SmartCartClustering.ipynb) uses the following flow:

1. Clean missing values.
2. Engineer features like age, total spending, total children, and customer tenure.
3. Encode categorical fields such as education and living situation.
4. Scale features.
5. Reduce dimensionality with PCA.
6. Predict clusters with agglomerative clustering.
7. Assign new uploads to the nearest trained cluster centroid in PCA space.

The Streamlit app in [app.py](app.py) loads the saved artifacts from [models/](models/) and reuses that exact logic for inference. K-Means is still available in the repository as a benchmark, but the website uses the agglomerative result for scoring.

For usability, the app supports three input modes:

1. Upload any CSV with partial or renamed columns.
2. Enter one customer manually in the form.
3. Download a template CSV that already has the expected fields.

## 👥 Typical users

- Marketing teams who want customer segments for targeted campaigns.
- Retail teams who want to identify high-spending or high-engagement groups.
- Data analysts who need a quick way to profile customers by behavior.
- Demo users who want to explore clustering results without running notebooks.

## 🗂️ Project structure

```text
SmartCart Clustering System/
├── app.py
├── Readme.MD
├── requirements.txt
├── data/
│   └── smartcart_customers.csv
├── models/
│   ├── aggl_model.pkl
│   ├── encoder.pkl
│   ├── kmeans_model.pkl
│   ├── pca.pkl
│   └── scaler.pkl
├── notebooks/
│   └── SmartCartClustering.ipynb
└── images/
```

## 📌 Files that matter

- [app.py](app.py): Streamlit dashboard and inference logic.
- [notebooks/SmartCartClustering.ipynb](notebooks/SmartCartClustering.ipynb): model training, feature engineering, and artifact export.
- [models/](models/): serialized scaler, encoder, PCA, and clustering model.
- [data/smartcart_customers.csv](data/smartcart_customers.csv): sample dataset for testing.
- [requirements.txt](requirements.txt): pinned Python dependencies.

## 🛠️ Local setup

1. Create a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Start the app.

```bash
streamlit run app.py
```

## 🧾 Input format

The uploaded CSV works best when it contains the same columns used by the training notebook, including:

- `ID`
- `Year_Birth`
- `Education`
- `Marital_Status`
- `Income`
- `Kidhome`
- `Teenhome`
- `Dt_Customer`
- `Recency`
- `MntWines`
- `MntFruits`
- `MntMeatProducts`
- `MntFishProducts`
- `MntSweetProducts`
- `MntGoldProds`
- `NumDealsPurchases`
- `NumWebPurchases`
- `NumCatalogPurchases`
- `NumStorePurchases`
- `NumWebVisitsMonth`
- `Complain`
- `Response`

## 📤 Output

The app returns:

- the original customer rows,
- a predicted `Cluster` column from the agglomerative model,
- a compact prediction view,
- a cluster count chart,
- a quick profile table,
- a downloadable clustered CSV.

## 🧰 Model artifacts

The app expects these files in [models/](models/):

- `aggl_model.pkl`
- `agglomerative_centroids.pkl`
- `scaler.pkl`
- `encoder.pkl`
- `pca.pkl`
- `feature_columns.pkl`

If you retrain the notebook, regenerate these artifacts before redeploying the app. The centroid file is required because agglomerative clustering does not provide a native `predict` method for new rows.

If your data uses different column names, the app will try to map them automatically and fill missing values with safe defaults.

## 🤖 K-Means vs Agglomerative

K-Means is still useful as a baseline because it is simple, fast, and easy to explain to non-technical users. In this project, however, agglomerative clustering is the better choice for predictions because it produced the stronger segmentation result and more useful clusters.

Quick comparison:

- K-Means: fast baseline, but it assumes spherical clusters and can miss more complex structure.
- Agglomerative clustering: better at capturing nested structure and producing more meaningful customer groups in this project.
- Deployment choice: agglomerative clustering is used in the app, with nearest-centroid assignment for incoming customer rows.

## 🌐 Deployment notes

- Use Streamlit Cloud, Azure App Service, or any host that supports Python apps.
- Make sure the `models/` folder is included in the deployed source.
- Keep [requirements.txt](requirements.txt) aligned with the artifact runtime.
- Include `agglomerative_centroids.pkl` with the other model files.
- Re-run the notebook if you change feature engineering logic.
- Mention the manual-entry and template tabs in your demo so users know the app is not limited to exact training files.

## 🔮 Future improvements

- Add a manual input form for single-customer scoring.
- Add cluster descriptions such as "high value" or "low engagement".
- Add more charts for spend and behavior comparison by cluster.
- Add a model retraining button or scheduled refresh pipeline.
- Add authentication if the app will be used internally.

## 📄 License

MIT or another license of your choice. Add a `LICENSE` file if you plan to publish the repository.
