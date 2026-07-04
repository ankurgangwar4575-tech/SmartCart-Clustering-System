from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


st.set_page_config(
	page_title="SmartCart Clustering System",
	page_icon="🛒",
	layout="wide",
)


st.markdown(
	"""
	<style>
	.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
	.hero {
		padding: 1.5rem 1.6rem;
		border-radius: 22px;
		background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #14b8a6 100%);
		color: white;
		box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
	}
	.hero h1 { margin: 0; font-size: 2.1rem; }
	.hero p { margin: 0.45rem 0 0; opacity: 0.92; }
	.card {
		padding: 1rem 1.1rem;
		border-radius: 16px;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
	}
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
	kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")
	scaler = joblib.load(MODEL_DIR / "scaler.pkl")
	encoder = joblib.load(MODEL_DIR / "encoder.pkl")
	pca_path = MODEL_DIR / "pca.pkl"
	if not pca_path.exists():
		raise FileNotFoundError(
			"Missing pca.pkl. Re-run the notebook after saving the PCA artifact."
		)
	pca = joblib.load(pca_path)
	return kmeans, scaler, encoder, pca


def clean_and_engineer(df: pd.DataFrame, encoder, scaler) -> pd.DataFrame:
	data = df.copy()

	required_columns = [
		"ID",
		"Year_Birth",
		"Education",
		"Marital_Status",
		"Income",
		"Kidhome",
		"Teenhome",
		"Dt_Customer",
		"Recency",
		"MntWines",
		"MntFruits",
		"MntMeatProducts",
		"MntFishProducts",
		"MntSweetProducts",
		"MntGoldProds",
		"NumDealsPurchases",
		"NumWebPurchases",
		"NumCatalogPurchases",
		"NumStorePurchases",
		"NumWebVisitsMonth",
		"Complain",
		"Response",
	]
	missing = [column for column in required_columns if column not in data.columns]
	if missing:
		raise ValueError(f"Missing required columns: {', '.join(missing)}")

	data["Income"] = pd.to_numeric(data["Income"], errors="coerce")
	data["Income"] = data["Income"].fillna(data["Income"].median())

	data["Year_Birth"] = pd.to_numeric(data["Year_Birth"], errors="coerce")
	data["Age"] = 2026 - data["Year_Birth"]

	data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True, errors="coerce")
	reference_date = data["Dt_Customer"].max()
	data["Customer_Tenure_Days"] = (reference_date - data["Dt_Customer"]).dt.days

	spending_columns = [
		"MntWines",
		"MntFruits",
		"MntMeatProducts",
		"MntFishProducts",
		"MntGoldProds",
		"MntSweetProducts",
	]
	data["Total_Spending"] = data[spending_columns].sum(axis=1)
	data["Total_Children"] = data["Kidhome"] + data["Teenhome"]

	data["Education"] = data["Education"].replace(
		{
			"Basic": "Undergraduate",
			"2n Cycle": "Undergraduate",
			"Graduation": "Graduate",
			"Master": "Postgraduate",
			"PhD": "Postgraduate",
			"Undegraduate": "Undergraduate",
		}
	)

	data["Living_With"] = data["Marital_Status"].replace(
		{
			"Married": "Partner",
			"Together": "Partner",
			"Single": "Alone",
			"Divorced": "Alone",
			"Widow": "Alone",
			"Absurd": "Alone",
			"YOLO": "Alone",
		}
	)

	drop_columns = [
		"ID",
		"Year_Birth",
		"Marital_Status",
		"Kidhome",
		"Teenhome",
		"Dt_Customer",
		*spending_columns,
	]
	cleaned = data.drop(columns=drop_columns, errors="ignore")

	cat_cols = ["Education", "Living_With"]
	encoded = encoder.transform(cleaned[cat_cols])
	encoded_df = pd.DataFrame(
		encoded.toarray() if hasattr(encoded, "toarray") else encoded,
		columns=encoder.get_feature_names_out(cat_cols),
		index=cleaned.index,
	)

	cleaned = pd.concat([cleaned.drop(columns=cat_cols), encoded_df], axis=1)
	feature_order = getattr(scaler, "feature_names_in_", cleaned.columns)
	cleaned = cleaned.reindex(columns=feature_order, fill_value=0)
	return cleaned


def predict_clusters(df: pd.DataFrame) -> pd.DataFrame:
	kmeans, scaler, encoder, pca = load_artifacts()
	processed = clean_and_engineer(df, encoder, scaler)
	scaled = scaler.transform(processed)
	reduced = pca.transform(scaled)
	labels = kmeans.predict(reduced)

	result = df.copy()
	result["Cluster"] = labels
	result["Age"] = 2026 - pd.to_numeric(result["Year_Birth"], errors="coerce")
	result["Total_Spending"] = (
		pd.to_numeric(result["MntWines"], errors="coerce").fillna(0)
		+ pd.to_numeric(result["MntFruits"], errors="coerce").fillna(0)
		+ pd.to_numeric(result["MntMeatProducts"], errors="coerce").fillna(0)
		+ pd.to_numeric(result["MntFishProducts"], errors="coerce").fillna(0)
		+ pd.to_numeric(result["MntGoldProds"], errors="coerce").fillna(0)
		+ pd.to_numeric(result["MntSweetProducts"], errors="coerce").fillna(0)
	)
	result["Total_Children"] = pd.to_numeric(result["Kidhome"], errors="coerce").fillna(0) + pd.to_numeric(result["Teenhome"], errors="coerce").fillna(0)
	return result


st.markdown(
	"""
	<div class="hero">
		<h1>SmartCart Clustering System</h1>
		<p>Upload customer records and assign cluster labels using the saved preprocessing and K-Means pipeline.</p>
	</div>
	""",
	unsafe_allow_html=True,
)


st.write("")
uploaded_file = st.file_uploader("Upload a customer CSV", type=["csv"])

if uploaded_file is None:
	st.info("Use the sample dataset in data/smartcart_customers.csv to test the deployment.")
	sample_path = ROOT / "data" / "smartcart_customers.csv"
	if sample_path.exists():
		sample_df = pd.read_csv(sample_path)
		st.dataframe(sample_df.head(), width="stretch")
	st.stop()


input_df = pd.read_csv(uploaded_file)

try:
	output_df = predict_clusters(input_df)
except Exception as exc:
	st.error(str(exc))
	st.stop()


cluster_counts = output_df["Cluster"].value_counts().sort_index()
total_rows = len(output_df)
summary_table = output_df.groupby("Cluster")[ ["Income", "Age", "Total_Spending", "Total_Children", "NumWebPurchases", "NumStorePurchases"] ].mean(numeric_only=True).round(2)

display_columns = [
	"Cluster",
	"Income",
	"Age",
	"Recency",
	"Total_Spending",
	"Total_Children",
	"NumWebPurchases",
	"NumStorePurchases",
	"NumCatalogPurchases",
	"NumDealsPurchases",
]
display_columns = [column for column in display_columns if column in output_df.columns]

col1, col2, col3 = st.columns(3)
col1.metric("Customers processed", f"{total_rows}")
col2.metric("Clusters found", f"{cluster_counts.shape[0]}")
col3.metric("Largest cluster", f"{int(cluster_counts.max()) if not cluster_counts.empty else 0}")


left, right = st.columns([1.1, 0.9])

with left:
	tab1, tab2 = st.tabs(["Prediction view", "Full data"])

	with tab1:
		st.subheader("Cluster assignments")
		st.dataframe(output_df[display_columns], width="stretch", hide_index=True, height=360)

	with tab2:
		st.subheader("Full uploaded dataset")
		st.dataframe(output_df, width="stretch", hide_index=True, height=360)

	csv_data = output_df.to_csv(index=False).encode("utf-8")
	st.download_button(
		"Download clustered CSV",
		data=csv_data,
		file_name="smartcart_clustered_customers.csv",
		mime="text/csv",
	)

with right:
	st.subheader("Cluster distribution")
	st.bar_chart(cluster_counts)

	st.subheader("Quick profile")
	st.dataframe(summary_table, width="stretch", hide_index=False)

