# 🛒 SmartCart Clustering System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/Machine%20Learning-K--Means-F7931E)](#how-it-works)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

SmartCart Clustering System is a Streamlit app for customer segmentation. It takes a customer CSV, applies the same preprocessing used during training, and assigns each record to a behavioral cluster. The goal is to help retail teams understand customer groups for targeting, retention, and campaign planning.

## ✨ What the app does

When a user opens the website, they can:

1. Upload a customer CSV.
2. See customer-level cluster labels.
3. Review cluster counts and a quick cluster profile.
4. Download the clustered output as a CSV.

This makes the project useful for marketing analysts, retail managers, and data teams who want a simple segmentation tool instead of working directly in notebooks.

## 🚀 Main features

- 📤 File upload support for customer CSVs.
- 🧠 Automatic preprocessing and feature engineering.
- 🧩 Cluster prediction using saved model artifacts.
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
6. Predict clusters with K-Means.

The Streamlit app in [app.py](app.py) loads the saved artifacts from [models/](models/) and reuses that exact logic for inference.

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

The uploaded CSV should contain the same columns used by the training notebook, including:

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
- a predicted `Cluster` column,
- a compact prediction view,
- a cluster count chart,
- a quick profile table,
- a downloadable clustered CSV.

## 🧰 Model artifacts

The app expects these files in [models/](models/):

- `kmeans_model.pkl`
- `scaler.pkl`
- `encoder.pkl`
- `pca.pkl`
- `feature_columns.pkl`

If you retrain the notebook, regenerate these artifacts before redeploying the app.

## 🤖 Why K-Means

K-Means is a good fit because it is simple, fast, and easy to explain to non-technical users. It works well as a baseline for customer segmentation when the goal is to create useful behavior groups quickly.

## 🌐 Deployment notes

- Use Streamlit Cloud, Azure App Service, or any host that supports Python apps.
- Make sure the `models/` folder is included in the deployed source.
- Keep [requirements.txt](requirements.txt) aligned with the artifact runtime.
- Re-run the notebook if you change feature engineering logic.

## 🔮 Future improvements

- Add a manual input form for single-customer scoring.
- Add cluster descriptions such as "high value" or "low engagement".
- Add more charts for spend and behavior comparison by cluster.
- Add a model retraining button or scheduled refresh pipeline.
- Add authentication if the app will be used internally.

## 📄 License

MIT or another license of your choice. Add a `LICENSE` file if you plan to publish the repository.
