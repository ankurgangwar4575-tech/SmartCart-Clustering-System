from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

MODEL_INPUT_FIELDS = [
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

BATCH_TEMPLATE_ROWS = 3

DEFAULT_INPUT_VALUES = {
	"ID": 1,
	"Year_Birth": 1970,
	"Education": "Graduation",
	"Marital_Status": "Married",
	"Income": 58000,
	"Kidhome": 0,
	"Teenhome": 0,
	"Dt_Customer": "2014-01-01",
	"Recency": 30,
	"MntWines": 100,
	"MntFruits": 20,
	"MntMeatProducts": 120,
	"MntFishProducts": 20,
	"MntSweetProducts": 10,
	"MntGoldProds": 15,
	"NumDealsPurchases": 2,
	"NumWebPurchases": 3,
	"NumCatalogPurchases": 2,
	"NumStorePurchases": 4,
	"NumWebVisitsMonth": 5,
	"Complain": 0,
	"Response": 0,
}

STRING_FIELDS = {"Education", "Marital_Status", "Dt_Customer"}
NUMERIC_FIELDS = [field for field in MODEL_INPUT_FIELDS if field not in STRING_FIELDS]

AUXILIARY_ALIASES = {
	"ID": ["id", "customerid", "customer_id"],
	"Year_Birth": ["year_birth", "yearofbirth", "birth_year", "birthyear", "dobyear"],
	"Education": ["education", "edu"],
	"Marital_Status": ["marital_status", "maritalstatus", "status"],
	"Income": ["income", "annualincome", "annual_income"],
	"Kidhome": ["kidhome", "kids", "childrenhome"],
	"Teenhome": ["teenhome", "teenshome"],
	"Dt_Customer": ["dt_customer", "customerdate", "join_date", "signup_date"],
	"Recency": ["recency", "days_since_last_purchase"],
	"MntWines": ["mntwines", "wines_spend"],
	"MntFruits": ["mntfruits", "fruits_spend"],
	"MntMeatProducts": ["mntmeatproducts", "meat_spend"],
	"MntFishProducts": ["mntfishproducts", "fish_spend"],
	"MntSweetProducts": ["mntsweetproducts", "sweet_spend"],
	"MntGoldProds": ["mntgoldprods", "gold_spend"],
	"NumDealsPurchases": ["numdealspurchases", "dealspurchases"],
	"NumWebPurchases": ["numwebpurchases", "webpurchases"],
	"NumCatalogPurchases": ["numcatalogpurchases", "catalogpurchases"],
	"NumStorePurchases": ["numstorepurchases", "storepurchases"],
	"NumWebVisitsMonth": ["numwebvisitsmonth", "webvisits"],
	"Complain": ["complain", "complaints"],
	"Response": ["response", "campaignresponse"],
}


def normalize_name(value: str) -> str:
	return "".join(character for character in str(value).lower() if character.isalnum())


def match_source_column(source_columns: list[str], target_field: str) -> str | None:
	normalized_source = {normalize_name(column): column for column in source_columns}
	if target_field in source_columns:
		return target_field
	for alias in [target_field, *AUXILIARY_ALIASES.get(target_field, [])]:
		matched = normalized_source.get(normalize_name(alias))
		if matched:
			return matched
	for source_column in source_columns:
		source_normalized = normalize_name(source_column)
		if normalize_name(target_field) in source_normalized or source_normalized in normalize_name(target_field):
			return source_column
	return None


def build_template_frame() -> pd.DataFrame:
	return pd.DataFrame([DEFAULT_INPUT_VALUES], columns=MODEL_INPUT_FIELDS)


def build_batch_template_frame(rows: int = BATCH_TEMPLATE_ROWS) -> pd.DataFrame:
	return pd.concat([build_template_frame() for _ in range(rows)], ignore_index=True)


def standardize_input_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
	source_columns = list(df.columns)
	standardized = pd.DataFrame(index=df.index)
	mapping = {}

	for field in MODEL_INPUT_FIELDS:
		source_column = match_source_column(source_columns, field)
		mapping[field] = source_column
		if source_column is None:
			standardized[field] = DEFAULT_INPUT_VALUES[field]
		else:
			standardized[field] = df[source_column]

	return standardized, mapping


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
	aggl_model = joblib.load(MODEL_DIR / "aggl_model.pkl")
	scaler = joblib.load(MODEL_DIR / "scaler.pkl")
	encoder = joblib.load(MODEL_DIR / "encoder.pkl")
	pca_path = MODEL_DIR / "pca.pkl"
	if not pca_path.exists():
		raise FileNotFoundError(
			"Missing pca.pkl. Re-run the notebook after saving the PCA artifact."
		)
	pca = joblib.load(pca_path)
	centroids_path = MODEL_DIR / "agglomerative_centroids.pkl"
	if not centroids_path.exists():
		raise FileNotFoundError(
			"Missing agglomerative_centroids.pkl. Rebuild the centroid artifact for agglomerative predictions."
		)
	centroid_artifact = joblib.load(centroids_path)
	return aggl_model, scaler, encoder, pca, centroid_artifact


def clean_and_engineer(df: pd.DataFrame, encoder, scaler) -> pd.DataFrame:
	data = df.copy()

	for field in MODEL_INPUT_FIELDS:
		if field not in data.columns:
			data[field] = DEFAULT_INPUT_VALUES[field]

	data = data[MODEL_INPUT_FIELDS]

	data["Income"] = pd.to_numeric(data["Income"], errors="coerce")
	data["Income"] = data["Income"].fillna(DEFAULT_INPUT_VALUES["Income"])

	data["Year_Birth"] = pd.to_numeric(data["Year_Birth"], errors="coerce")
	data["Year_Birth"] = data["Year_Birth"].fillna(DEFAULT_INPUT_VALUES["Year_Birth"])
	data["Age"] = 2026 - data["Year_Birth"]

	data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True, errors="coerce")
	data["Dt_Customer"] = data["Dt_Customer"].fillna(pd.Timestamp(DEFAULT_INPUT_VALUES["Dt_Customer"]))
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
	aggl_model, scaler, encoder, pca, centroid_artifact = load_artifacts()
	processed = clean_and_engineer(df, encoder, scaler)
	scaled = scaler.transform(processed)
	reduced = pca.transform(scaled)
	cluster_ids = np.asarray(centroid_artifact["cluster_ids"])
	centroids = np.asarray(centroid_artifact["centroids"])
	distances = np.linalg.norm(reduced[:, None, :] - centroids[None, :, :], axis=2)
	nearest = distances.argmin(axis=1)
	labels = cluster_ids[nearest]

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


def predict_from_source_frame(source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	standardized_df, mapping = standardize_input_frame(source_df)
	prediction_df = predict_clusters(standardized_df)
	prediction_df = pd.concat([source_df.reset_index(drop=True), prediction_df[["Cluster", "Age", "Total_Spending", "Total_Children"]].reset_index(drop=True)], axis=1)
	mapping_df = pd.DataFrame(
		{
			"Model field": MODEL_INPUT_FIELDS,
			"Source column": [mapping[field] if mapping[field] is not None else "Default value" for field in MODEL_INPUT_FIELDS],
		}
	)
	return prediction_df, mapping_df


def render_results(output_df: pd.DataFrame, key_prefix: str) -> None:
	cluster_counts = output_df["Cluster"].value_counts().sort_index()
	total_rows = len(output_df)
	summary_table = output_df.groupby("Cluster")[["Income", "Age", "Total_Spending", "Total_Children", "NumWebPurchases", "NumStorePurchases"]].mean(numeric_only=True).round(2)

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
			key=f"{key_prefix}_download_csv",
		)

	with right:
		st.subheader("Cluster distribution")
		st.bar_chart(cluster_counts)

		st.subheader("Quick profile")
		st.dataframe(summary_table, width="stretch", hide_index=False)


st.markdown(
	"""
	<div class="hero">
		<h1>SmartCart Clustering System</h1>
		<p>Upload any customer table or enter a single customer manually, then assign cluster labels using the agglomerative clustering pipeline.</p>
	</div>
	""",
	unsafe_allow_html=True,
)

tab_upload, tab_manual, tab_template = st.tabs(["Upload CSV", "Manual Entry", "Template"])

with tab_upload:
	st.write("Upload any customer CSV. The app will auto-match known columns and use defaults for anything missing.")
	uploaded_file = st.file_uploader("Upload a customer CSV", type=["csv"], key="csv_upload")

	if uploaded_file is not None:
		raw_df = pd.read_csv(uploaded_file)
		try:
			output_df, mapping_df = predict_from_source_frame(raw_df)
		except Exception as exc:
			st.error(str(exc))
		else:
			with st.expander("Detected column mapping", expanded=False):
				st.dataframe(mapping_df, width="stretch", hide_index=True)
			render_results(output_df, key_prefix="upload")
	else:
		sample_path = ROOT / "data" / "smartcart_customers.csv"
		if sample_path.exists():
			st.info("Use the sample dataset in data/smartcart_customers.csv to test the deployment.")
			sample_df = pd.read_csv(sample_path)
			st.dataframe(sample_df.head(), width="stretch")

with tab_manual:
	st.write("Enter one customer record and predict its cluster.")
	with st.form("manual_customer_form"):
		left_form, right_form = st.columns(2)

		with left_form:
			customer_id = st.number_input("ID", min_value=1, value=int(DEFAULT_INPUT_VALUES["ID"]), step=1)
			year_birth = st.number_input("Year of Birth", min_value=1920, max_value=2006, value=int(DEFAULT_INPUT_VALUES["Year_Birth"]), step=1)
			education = st.selectbox("Education", ["Basic", "2n Cycle", "Graduation", "Master", "PhD"], index=2)
			marital_status = st.selectbox("Marital Status", ["Married", "Together", "Single", "Divorced", "Widow", "Absurd", "YOLO"], index=0)
			income = st.number_input("Income", min_value=0.0, value=float(DEFAULT_INPUT_VALUES["Income"]), step=1000.0)
			kidhome = st.number_input("Kidhome", min_value=0, max_value=10, value=int(DEFAULT_INPUT_VALUES["Kidhome"]), step=1)
			teenhome = st.number_input("Teenhome", min_value=0, max_value=10, value=int(DEFAULT_INPUT_VALUES["Teenhome"]), step=1)
			dt_customer = st.date_input("Customer Join Date")
			recency = st.number_input("Recency", min_value=0, value=int(DEFAULT_INPUT_VALUES["Recency"]), step=1)
			mnt_wines = st.number_input("MntWines", min_value=0, value=int(DEFAULT_INPUT_VALUES["MntWines"]), step=1)
			mnt_fruits = st.number_input("MntFruits", min_value=0, value=int(DEFAULT_INPUT_VALUES["MntFruits"]), step=1)
			mnt_meat = st.number_input("MntMeatProducts", min_value=0, value=int(DEFAULT_INPUT_VALUES["MntMeatProducts"]), step=1)

		with right_form:
			mnt_fish = st.number_input("MntFishProducts", min_value=0, value=int(DEFAULT_INPUT_VALUES["MntFishProducts"]), step=1)
			mnt_sweet = st.number_input("MntSweetProducts", min_value=0, value=int(DEFAULT_INPUT_VALUES["MntSweetProducts"]), step=1)
			mnt_gold = st.number_input("MntGoldProds", min_value=0, value=int(DEFAULT_INPUT_VALUES["MntGoldProds"]), step=1)
			deals = st.number_input("NumDealsPurchases", min_value=0, value=int(DEFAULT_INPUT_VALUES["NumDealsPurchases"]), step=1)
			web = st.number_input("NumWebPurchases", min_value=0, value=int(DEFAULT_INPUT_VALUES["NumWebPurchases"]), step=1)
			catalog = st.number_input("NumCatalogPurchases", min_value=0, value=int(DEFAULT_INPUT_VALUES["NumCatalogPurchases"]), step=1)
			store = st.number_input("NumStorePurchases", min_value=0, value=int(DEFAULT_INPUT_VALUES["NumStorePurchases"]), step=1)
			visits = st.number_input("NumWebVisitsMonth", min_value=0, value=int(DEFAULT_INPUT_VALUES["NumWebVisitsMonth"]), step=1)
			complain = st.number_input("Complain", min_value=0, max_value=1, value=int(DEFAULT_INPUT_VALUES["Complain"]), step=1)
			response = st.number_input("Response", min_value=0, max_value=1, value=int(DEFAULT_INPUT_VALUES["Response"]), step=1)

		submit = st.form_submit_button("Predict cluster")

	if submit:
		manual_df = pd.DataFrame(
			[
				{
					"ID": customer_id,
					"Year_Birth": year_birth,
					"Education": education,
					"Marital_Status": marital_status,
					"Income": income,
					"Kidhome": kidhome,
					"Teenhome": teenhome,
					"Dt_Customer": dt_customer.strftime("%Y-%m-%d"),
					"Recency": recency,
					"MntWines": mnt_wines,
					"MntFruits": mnt_fruits,
					"MntMeatProducts": mnt_meat,
					"MntFishProducts": mnt_fish,
					"MntSweetProducts": mnt_sweet,
					"MntGoldProds": mnt_gold,
					"NumDealsPurchases": deals,
					"NumWebPurchases": web,
					"NumCatalogPurchases": catalog,
					"NumStorePurchases": store,
					"NumWebVisitsMonth": visits,
					"Complain": complain,
					"Response": response,
				}
			]
		)
		try:
			output_df = predict_clusters(manual_df)
		except Exception as exc:
			st.error(str(exc))
		else:
			render_results(output_df, key_prefix="manual")

	st.divider()
	st.write("Need more than one manual prediction? Edit a small batch below and predict all rows at once.")
	manual_batch_df = st.data_editor(
		build_batch_template_frame(),
		num_rows="dynamic",
		width="stretch",
		key="manual_batch_editor",
	)

	if st.button("Predict manual batch", key="predict_manual_batch"):
		try:
			output_df, mapping_df = predict_from_source_frame(manual_batch_df)
		except Exception as exc:
			st.error(str(exc))
		else:
			with st.expander("Detected column mapping", expanded=False):
				st.dataframe(mapping_df, width="stretch", hide_index=True)
			render_results(output_df, key_prefix="manual_batch")

with tab_template:
	st.write("Download a starter CSV with the exact fields the model knows how to use. This template is useful when your source data uses different names or when you want to prepare a file manually.")
	template_df = build_template_frame()
	st.dataframe(template_df, width="stretch", hide_index=True)
	st.download_button(
		"Download template CSV",
		data=template_df.to_csv(index=False).encode("utf-8"),
		file_name="smartcart_template.csv",
		mime="text/csv",
		key="template_download_csv",
	)

