import streamlit as st
import pandas as pd
import numpy as np
import pycountry
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction",
    layout="wide",
    page_icon="💼"
)

# HEADER
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💼 Cybersecurity Salary Prediction</h1>', unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("salaries_cyber_clean.csv")

df = load_data()

# ---------------------------------------------------------
# COUNTRY NAME EXPANSION
# ---------------------------------------------------------
def expand_country(code):
    try:
        country = pycountry.countries.get(alpha_2=code)
        return f"{code} — {country.name}" if country else f"{code} — Unknown"
    except:
        return f"{code} — Unknown"

company_location_map = {c: expand_country(c) for c in df["company_location"].unique()}
employee_residence_map = {c: expand_country(c) for c in df["employee_residence"].unique()}

# ---------------------------------------------------------
# LABEL MAPPINGS
# ---------------------------------------------------------
experience_map = {
    "EN": "EN — Entry-level / Junior",
    "MI": "MI — Mid-level / Intermediate",
    "SE": "SE — Senior-level",
    "EX": "EX — Executive / Director"
}

employment_map = {
    "FT": "FT — Full-time",
    "PT": "PT — Part-time",
    "CT": "CT — Contract",
    "FL": "FL — Freelance"
}

size_map = {"S": "S — Small", "M": "M — Medium", "L": "L — Large"}

# ---------------------------------------------------------
# TRAIN RANDOM FOREST MODEL (LOG TARGET)
# ---------------------------------------------------------
FEATURES = [
    "job_title",
    "experience_level",
    "employment_type",
    "company_location",
    "company_size",
    "employee_residence",
    "remote_ratio"
]

@st.cache_resource
def train_final_rf(data):

    X = data[FEATURES]
    y_log = np.log1p(data["salary_in_usd"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)],
        remainder="passthrough"
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("rf", rf)
    ])

    model.fit(X_train, y_train)
    return model

rf_model = train_final_rf(df)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("⚙️ Model Configuration")

st.sidebar.subheader("🔧 Prediction Method")
st.sidebar.info("Random Forest (Log Target + OneHot Encoding)")

st.sidebar.subheader("📘 Dataset Info")
st.sidebar.markdown(f"""
- **Total Records:** {len(df):,}  
- **Years Covered:** 2020–2022  
- **Unique Job Titles:** {df['job_title'].nunique()}  
- **Average Salary:** ${df['salary_in_usd'].mean():,.0f}  
""")

st.sidebar.subheader("📊 Model Performance")
st.sidebar.success("**MSE:** 0.198034")
st.sidebar.success("**RMSE:** 0.445010")
st.sidebar.success("**R²:** 0.527677")

# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔮 Predict Cybersecurity Salary")

col1, col2, col3 = st.columns(3)
with col1:
    job = st.selectbox("👔 Job Title", sorted(df["job_title"].unique()))

with col2:
    exp_label = st.selectbox(
        "📈 Experience Level",
        [experience_map[e] for e in sorted(experience_map.keys())]
    )
    exp = exp_label.split(" — ")[0]

with col3:
    emp_type_label = st.selectbox(
        "💼 Employment Type",
        [employment_map[e] for e in sorted(employment_map.keys())]
    )
    emp_type = emp_type_label.split(" — ")[0]

col4, col5, col6 = st.columns(3)
with col4:
    comp_loc_label = st.selectbox(
        "🌍 Company Location",
        [company_location_map[c] for c in sorted(company_location_map.keys())]
    )
    comp_loc = comp_loc_label.split(" — ")[0]

with col5:
    emp_res_label = st.selectbox(
        "🏡 Employee Residence",
        [employee_residence_map[c] for c in sorted(employee_residence_map.keys())]
    )
    emp_res = emp_res_label.split(" — ")[0]

with col6:
    remote = st.selectbox("🧑‍💻 Remote Ratio (%)", sorted(df["remote_ratio"].unique()))

size_label = st.selectbox(
    "🏢 Company Size",
    [size_map[s] for s in sorted(size_map.keys())]
)
company_size = size_label.split(" — ")[0]

# ---------------------------------------------------------
# AUTO PREDICTION
# ---------------------------------------------------------
user_input = pd.DataFrame([{
    "job_title": job,
    "experience_level": exp,
    "employment_type": emp_type,
    "company_location": comp_loc,
    "company_size": company_size,
    "employee_residence": emp_res,
    "remote_ratio": remote
}])

log_pred = rf_model.predict(user_input)[0]
salary_pred = np.expm1(log_pred)

# DISPLAY RESULT (Gradient Highlight Box)
st.markdown(f"""
<div style="
    padding: 22px;
    border-radius: 14px;
    background: linear-gradient(135deg, #667eea20, #764ba220);
    border-left: 6px solid #667eea;
    box-shadow: 0 3px 10px rgba(0,0,0,0.12);
    margin-bottom: 25px;
">
    <h3 style="margin: 0; font-weight: 600;">💰 Predicted Salary</h3>
    <p style="font-size: 2rem; font-weight: bold; margin-top: 8px;">
        ${salary_pred:,.2f}
    </p>
</div>
""", unsafe_allow_html=True)



# ---------------------------------------------------------
# SALARY DISTRIBUTION COMPARISON
# ---------------------------------------------------------
st.subheader("📊 Salary Distribution Comparison")

st.caption("This chart compares your predicted salary with the real salary distribution in the dataset. The red line shows your predicted value.")

fig_dist = go.Figure()

fig_dist.add_trace(go.Histogram(
    x=df["salary_in_usd"],
    nbinsx=40,
    marker=dict(color="#667eea"),
    opacity=0.75,
    name="Training Salary Distribution"
))

fig_dist.add_vline(
    x=salary_pred,
    line_width=3,
    line_color="red",
    annotation_text="Predicted Salary",
    annotation_position="top"
)

fig_dist.update_layout(
    xaxis_title="Salary (USD)",
    yaxis_title="Count",
    template="plotly_white",
    height=450
)

st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------
# FEATURE IMPORTANCE (FIXED)
# ---------------------------------------------------------
st.subheader("📌 Feature Importance (Random Forest)")
st.caption("This chart shows which input features influenced the salary prediction the most.")

rf = rf_model.named_steps["rf"]
ohe = rf_model.named_steps["prep"].named_transformers_["cat"]

# Number of OHE outputs per feature
category_sizes = [len(ohe.categories_[i]) for i in range(len(FEATURES))]

# Aggregate importances
importances = rf.feature_importances_
importance_dict = {}
index = 0
for feat, size in zip(FEATURES, category_sizes):
    importance_dict[feat] = importances[index:index+size].sum()
    index += size

# Plot importances
imp_df = pd.DataFrame({
    "feature": list(importance_dict.keys()),
    "importance": list(importance_dict.values())
}).sort_values("importance", ascending=False)

fig_imp = go.Figure()

fig_imp.add_trace(go.Bar(
    x=imp_df["importance"],
    y=imp_df["feature"],
    orientation="h",
    marker=dict(
        color=imp_df["importance"],
        colorscale="Blues",
        line=dict(color="black", width=1)
    )
))

fig_imp.update_layout(
    xaxis_title="Importance Score",
    yaxis_title="Feature",
    template="plotly_white",
    height=450
)

st.plotly_chart(fig_imp, use_container_width=True)
