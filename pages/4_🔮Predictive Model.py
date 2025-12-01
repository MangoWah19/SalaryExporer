import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide",
    page_icon="💼"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💼 Cybersecurity Salary Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("*Predict salaries with advanced ML models and interactive visualizations*")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    return df

df = load_data()

# ---------------------------------------------------------
# Sidebar - Model Selection & Info
# ---------------------------------------------------------
st.sidebar.header("🎛️ Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"],
    help="Choose the prediction algorithm"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.info(f"""
- **Total Records:** {len(df):,}
- **Date Range:** {df['work_year'].min()} - {df['work_year'].max()}
- **Job Titles:** {df['job_title'].nunique()}
- **Avg Salary:** ${df['salary_in_usd'].mean():,.0f}
""")

# Show current model being used
st.sidebar.success(f"**Active Model:** {model_type}")

# ---------------------------------------------------------
# Train Models with Performance Metrics
# ---------------------------------------------------------
@st.cache_resource
def train_models(data):
    X = data[["work_year", "job_title", "experience_level", "company_size"]]
    y = data["salary_in_usd"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_cols = ["job_title", "experience_level", "company_size"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
        remainder="passthrough"
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    trained_models = {}
    metrics = {}
    
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        trained_models[name] = pipeline
        metrics[name] = {
            "R² Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    return trained_models, metrics

models, performance_metrics = train_models(df)
selected_model = models[model_type]

# ---------------------------------------------------------
# Display Model Performance
# ---------------------------------------------------------
with st.sidebar.expander("📈 Model Performance"):
    perf = performance_metrics[model_type]
    st.metric("R² Score", f"{perf['R² Score']:.3f}")
    st.metric("MAE", f"${perf['MAE']:,.0f}")
    st.metric("RMSE", f"${perf['RMSE']:,.0f}")

# ---------------------------------------------------------
# Main Content - Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Salary Forecast", "📊 Data Analysis", "💰 Salary Calculator", "🗺️ Insights"])

# ---------------------------------------------------------
# TAB 1: Salary Forecast
# ---------------------------------------------------------
with tab1:
    st.subheader("⚙️ Customize Your Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        custom_job = st.selectbox("👔 Job Title", sorted(df["job_title"].unique()))
    with col2:
        custom_exp = st.selectbox("📈 Experience Level", sorted(df["experience_level"].unique()))
    with col3:
        custom_size = st.selectbox("🏢 Company Size", sorted(df["company_size"].unique()))
    
    # Forecast 2020–2030 (actual data 2020-2022, predictions 2023-2030)
    all_forecast_years = np.arange(2020, 2031)
    forecast_data = []
    
    for year in all_forecast_years:
        if year <= 2022:
            # Try to get actual data with exact match
            actual_data = df[
                (df["work_year"] == year) & 
                (df["job_title"] == custom_job) & 
                (df["experience_level"] == custom_exp) & 
                (df["company_size"] == custom_size)
            ]
            
            if len(actual_data) > 0:
                salary = actual_data["salary_in_usd"].mean()
                data_source = "Actual"
            else:
                # Try without company size
                actual_data_partial = df[
                    (df["work_year"] == year) & 
                    (df["job_title"] == custom_job) & 
                    (df["experience_level"] == custom_exp)
                ]
                
                if len(actual_data_partial) > 0:
                    salary = actual_data_partial["salary_in_usd"].mean()
                    data_source = "Actual (Partial Match)"
                else:
                    # Try with just job title
                    actual_data_job = df[
                        (df["work_year"] == year) & 
                        (df["job_title"] == custom_job)
                    ]
                    
                    if len(actual_data_job) > 0:
                        salary = actual_data_job["salary_in_usd"].mean()
                        data_source = "Actual (Job Only)"
                    else:
                        # Use prediction if no match at all
                        pred_input = pd.DataFrame({
                            "work_year": [year],
                            "job_title": [custom_job],
                            "experience_level": [custom_exp],
                            "company_size": [custom_size]
                        })
                        salary = selected_model.predict(pred_input)[0]
                        data_source = "Predicted"
        else:
            # Predict for 2023-2030
            pred_input = pd.DataFrame({
                "work_year": [year],
                "job_title": [custom_job],
                "experience_level": [custom_exp],
                "company_size": [custom_size]
            })
            salary = selected_model.predict(pred_input)[0]
            data_source = "Predicted"
        
        forecast_data.append({
            "Year": year,
            "Salary (USD)": salary,
            "Source": data_source
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Display data source information
    st.info(f"""
    **Data Sources Used:**
    - Years with Actual Data: {', '.join(map(str, forecast_df[forecast_df['Source'].str.contains('Actual')]['Year'].tolist()))}
    - Years with Predicted Data: {', '.join(map(str, forecast_df[forecast_df['Source'] == 'Predicted']['Year'].tolist()))}
    """)
    
    # Enhanced Graph
    st.markdown("---")
    
    # Calculate metrics
    actual_data_count = len(forecast_df[forecast_df["Source"] == "Actual"])
    start_salary = forecast_df.iloc[0]['Salary (USD)']
    end_salary = forecast_df.iloc[-1]['Salary (USD)']
    current_2025 = forecast_df[forecast_df["Year"] == 2025].iloc[0]['Salary (USD)']
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("📅 Starting Salary (2020)", f"${start_salary:,.0f}")
    with col_b:
        st.metric("📅 Current Salary (2025)", f"${current_2025:,.0f}")
    with col_c:
        growth = ((end_salary - start_salary) / start_salary) * 100
        st.metric("📈 10-Year Growth", f"{growth:.1f}%")
    
    # Create visualization with color distinction
    fig = go.Figure()
    
    # Actual data (any year with actual data)
    actual_df = forecast_df[forecast_df["Source"].str.contains("Actual")]
    if len(actual_df) > 0:
        fig.add_trace(go.Scatter(
            x=actual_df["Year"],
            y=actual_df["Salary (USD)"],
            mode='lines+markers',
            name='Actual Data',
            line=dict(color='#10b981', width=4),
            marker=dict(size=12, color='#10b981', line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)',
            hovertemplate='Year: %{x}<br>Salary: $%{y:,.0f}<br>Source: Actual<extra></extra>'
        ))
    
    # Predicted data
    predicted_df = forecast_df[forecast_df["Source"] == "Predicted"]
    if len(predicted_df) > 0:
        fig.add_trace(go.Scatter(
            x=predicted_df["Year"],
            y=predicted_df["Salary (USD)"],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#667eea', width=4, dash='dash'),
            marker=dict(size=12, color='#764ba2', line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)',
            hovertemplate='Year: %{x}<br>Salary: $%{y:,.0f}<br>Source: Predicted<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Salary Forecast (2020-2030): {custom_job} ({custom_exp}, {custom_size})",
        xaxis_title="Year",
        yaxis_title="Salary (USD)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        xaxis=dict(dtick=1),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Data Analysis
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Salary Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary by Experience Level
        exp_avg = df.groupby("experience_level")["salary_in_usd"].mean().reset_index()
        fig_exp = px.bar(
            exp_avg,
            x="experience_level",
            y="salary_in_usd",
            title="Average Salary by Experience Level",
            color="salary_in_usd",
            color_continuous_scale="Viridis"
        )
        fig_exp.update_layout(showlegend=False, xaxis_title="Experience Level", yaxis_title="Avg Salary (USD)")
        st.plotly_chart(fig_exp, use_container_width=True)
    
    with col2:
        # Salary by Company Size
        size_avg = df.groupby("company_size")["salary_in_usd"].mean().reset_index()
        fig_size = px.bar(
            size_avg,
            x="company_size",
            y="salary_in_usd",
            title="Average Salary by Company Size",
            color="salary_in_usd",
            color_continuous_scale="Plasma"
        )
        fig_size.update_layout(showlegend=False, xaxis_title="Company Size", yaxis_title="Avg Salary (USD)")
        st.plotly_chart(fig_size, use_container_width=True)
    
    # Top Paying Jobs 2021-2030 with Year Selector
    st.markdown("---")
    st.subheader("💎 Top 10 Highest Paying Jobs by Year")
    
    # Year selector for top paying jobs
    selected_year_top = st.selectbox(
        "📅 Select Year to View Top Jobs",
        list(range(2021, 2031)),
        index=4,  # Default to 2025
        key="top_jobs_year"
    )
    
    # Get all unique job titles
    all_jobs = df["job_title"].unique()
    job_salaries_for_year = []
    
    for job in all_jobs:
        # Use actual data for 2020-2022, try actual for 2023-2025, otherwise predict
        if selected_year_top <= 2025:
            actual_data = df[(df["job_title"] == job) & (df["work_year"] == selected_year_top)]
            if len(actual_data) > 0:
                avg_salary = actual_data["salary_in_usd"].mean()
                job_salaries_for_year.append({
                    "job_title": job,
                    "salary": avg_salary
                })
            elif selected_year_top >= 2023:
                # For 2023-2025, if no actual data found, predict
                job_data = df[df["job_title"] == job]
                if len(job_data) > 0:
                    most_common_exp = job_data["experience_level"].mode()[0]
                    most_common_size = job_data["company_size"].mode()[0]
                    
                    pred_input = pd.DataFrame({
                        "work_year": [selected_year_top],
                        "job_title": [job],
                        "experience_level": [most_common_exp],
                        "company_size": [most_common_size]
                    })
                    
                    predicted_salary = selected_model.predict(pred_input)[0]
                    job_salaries_for_year.append({
                        "job_title": job,
                        "salary": predicted_salary
                    })
        else:
            # Predict for 2026-2030
            job_data = df[df["job_title"] == job]
            if len(job_data) > 0:
                most_common_exp = job_data["experience_level"].mode()[0]
                most_common_size = job_data["company_size"].mode()[0]
                
                pred_input = pd.DataFrame({
                    "work_year": [selected_year_top],
                    "job_title": [job],
                    "experience_level": [most_common_exp],
                    "company_size": [most_common_size]
                })
                
                predicted_salary = selected_model.predict(pred_input)[0]
                job_salaries_for_year.append({
                    "job_title": job,
                    "salary": predicted_salary
                })
    
    # Convert to DataFrame and get top 10
    if len(job_salaries_for_year) > 0:
        top_jobs_year = pd.DataFrame(job_salaries_for_year).sort_values("salary", ascending=False).head(10)
    else:
        st.warning(f"⚠️ No data available for {selected_year_top}. Please select a different year.")
        top_jobs_year = pd.DataFrame(columns=["job_title", "salary"])
    
    # Determine if this year uses actual or predicted data
    if selected_year_top <= 2022:
        data_type = "Actual Data"
    elif selected_year_top <= 2025:
        # For 2023-2025, might be mix of actual and predicted
        data_type = "Actual/Predicted Data"
    else:
        data_type = "Predicted Data"
    
    if len(top_jobs_year) > 0:
        fig_top = px.bar(
            top_jobs_year,
            x="salary",
            y="job_title",
            orientation='h',
            title=f"Top 10 Highest Paying Jobs in {selected_year_top} ({data_type})",
            color="salary",
            color_continuous_scale="Turbo",
            labels={"salary": "Avg Salary (USD)", "job_title": "Job Title"}
        )
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, xaxis_title="Avg Salary (USD)", yaxis_title="")
        st.plotly_chart(fig_top, use_container_width=True)
    
    # Show salary comparison across years for top job
    st.markdown("---")
    st.subheader("📈 Salary Trends for Top 3 Highest Paying Jobs")
    
    # Get top 3 jobs from current selected year
    if len(top_jobs_year) >= 3:
        top_3_jobs = top_jobs_year.head(3)["job_title"].tolist()
    elif len(top_jobs_year) > 0:
        top_3_jobs = top_jobs_year["job_title"].tolist()
    else:
        st.info("ℹ️ Select a year with available data to view salary trends.")
        top_3_jobs = []
    
    if len(top_3_jobs) > 0:
        # Prepare data for all years (2020-2030) for these top 3 jobs
        all_years = np.arange(2020, 2031)
        job_salary_trends = []
        
        for job in top_3_jobs:
            job_data_full = df[df["job_title"] == job]
            if len(job_data_full) > 0:
                most_common_exp = job_data_full["experience_level"].mode()[0]
                most_common_size = job_data_full["company_size"].mode()[0]
                
                for year in all_years:
                    if year <= 2022:
                        # Use actual data for 2020-2022
                        actual_data = df[(df["job_title"] == job) & (df["work_year"] == year)]
                        if len(actual_data) > 0:
                            salary = actual_data["salary_in_usd"].mean()
                            data_type_point = "Actual"
                        else:
                            # If no actual data, predict
                            pred_input = pd.DataFrame({
                                "work_year": [year],
                                "job_title": [job],
                                "experience_level": [most_common_exp],
                                "company_size": [most_common_size]
                            })
                            salary = selected_model.predict(pred_input)[0]
                            data_type_point = "Predicted"
                    else:
                        # Predict for 2023-2030 (no reliable actual data available)
                        pred_input = pd.DataFrame({
                            "work_year": [year],
                            "job_title": [job],
                            "experience_level": [most_common_exp],
                            "company_size": [most_common_size]
                        })
                        salary = selected_model.predict(pred_input)[0]
                        data_type_point = "Predicted"
                    
                    job_salary_trends.append({
                        "job_title": job,
                        "year": year,
                        "salary": salary,
                        "type": data_type_point
                    })
        
        trends_df = pd.DataFrame(job_salary_trends)
        
        fig_trends = px.line(
            trends_df,
            x="year",
            y="salary",
            color="job_title",
            markers=True,
            title="Salary Trends (2020-2030) - Top 3 Jobs",
            labels={"year": "Year", "salary": "Salary (USD)", "job_title": "Job Title"}
        )
        
        # Add a vertical line to separate actual from predicted
        fig_trends.add_vline(x=2022.5, line_dash="dash", line_color="gray", 
                             annotation_text="Actual | Predicted", 
                             annotation_position="top")
        
        fig_trends.update_layout(hovermode="x unified", height=500)
        st.plotly_chart(fig_trends, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Salary Calculator
# ---------------------------------------------------------
with tab3:
    st.subheader("💰 Salary Calculator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        calc_year = st.slider("📅 Select Year", 2020, 2035, 2025)
    
    # Determine if we're using actual data or prediction
    is_actual_data = (calc_year >= 2020 and calc_year <= 2025)
    
    if is_actual_data:
        # Try to get actual data
        actual_calc_data = df[
            (df["work_year"] == calc_year) & 
            (df["job_title"] == custom_job) & 
            (df["experience_level"] == custom_exp) & 
            (df["company_size"] == custom_size)
        ]
        
        if len(actual_calc_data) > 0:
            # Use actual average
            salary_value = actual_calc_data["salary_in_usd"].mean()
            salary_label = "Actual Average Salary"
            data_badge = "📊 Actual Data"
        else:
            # No actual data available, use prediction
            single_input = pd.DataFrame({
                "work_year": [calc_year],
                "job_title": [custom_job],
                "experience_level": [custom_exp],
                "company_size": [custom_size]
            })
            salary_value = selected_model.predict(single_input)[0]
            salary_label = "Predicted Salary"
            data_badge = "🔮 Predicted (No Actual Data)"
    else:
        # Use prediction for years outside 2020-2022
        single_input = pd.DataFrame({
            "work_year": [calc_year],
            "job_title": [custom_job],
            "experience_level": [custom_exp],
            "company_size": [custom_size]
        })
        salary_value = selected_model.predict(single_input)[0]
        salary_label = "Predicted Salary"
        data_badge = "🔮 Predicted Data"
    
    # Display prediction with styling
    st.markdown("---")
    
    # Data type badge
    if is_actual_data and len(actual_calc_data) > 0:
        st.success(f"✅ {data_badge}")
    else:
        st.info(f"ℹ️ {data_badge}")
    
    col_x, col_y, col_z = st.columns([1, 2, 1])
    
    with col_y:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{salary_label} for {calc_year}</h3>
            <h1 style="font-size: 3rem; margin: 1rem 0;">${salary_value:,.0f}</h1>
            <p>{custom_job} | {custom_exp} | {custom_size}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison with market
    st.markdown("---")
    st.subheader("📊 Market Comparison")
    
    # Get market average for the selected year and job
    if calc_year >= 2020 and calc_year <= 2022:
        # For years with actual data, use year-specific market average
        market_data = df[(df["job_title"] == custom_job) & (df["work_year"] == calc_year)]
        if len(market_data) > 0:
            market_avg = market_data["salary_in_usd"].mean()
            market_label = f"Market Avg ({calc_year})"
        else:
            # If no data for this specific year, use overall average for this job
            market_avg = df[df["job_title"] == custom_job]["salary_in_usd"].mean()
            market_label = "Overall Market Avg"
    else:
        # For future years (2023+), predict market average for the job
        # Use median experience level and company size for market average
        job_data = df[df["job_title"] == custom_job]
        if len(job_data) > 0:
            # Get most common experience and company size for this job
            common_exp = job_data["experience_level"].mode()[0]
            common_size = job_data["company_size"].mode()[0]
            
            market_pred_input = pd.DataFrame({
                "work_year": [calc_year],
                "job_title": [custom_job],
                "experience_level": [common_exp],
                "company_size": [common_size]
            })
            market_avg = selected_model.predict(market_pred_input)[0]
            market_label = f"Predicted Market Avg ({calc_year})"
        else:
            market_avg = df[df["job_title"] == custom_job]["salary_in_usd"].mean()
            market_label = "Overall Market Avg"
    
    diff = salary_value - market_avg
    diff_pct = (diff / market_avg) * 100 if market_avg > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Your Salary", f"${salary_value:,.0f}")
    with col2:
        st.metric(market_label, f"${market_avg:,.0f}")
    with col3:
        st.metric("Difference", f"${diff:,.0f}", f"{diff_pct:+.1f}%")
    
    # Add explanation
    if calc_year >= 2023:
        st.caption(f"💡 Market average for {calc_year} is predicted using the most common experience level ({common_exp}) and company size ({common_size}) for {custom_job}.")

# ---------------------------------------------------------
# TAB 4: Insights
# ---------------------------------------------------------
with tab4:
    st.subheader("🗺️ Key Insights & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Yearly Salary Trends")
        yearly_avg = df.groupby("work_year")["salary_in_usd"].mean().reset_index()
        fig_yearly = px.line(
            yearly_avg,
            x="work_year",
            y="salary_in_usd",
            markers=True,
            title="Average Salary Over Years"
        )
        fig_yearly.update_traces(line=dict(color='#667eea', width=3), marker=dict(size=10))
        st.plotly_chart(fig_yearly, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Salary Distribution")
        fig_dist = px.histogram(
            df,
            x="salary_in_usd",
            nbins=50,
            title="Overall Salary Distribution",
            color_discrete_sequence=['#764ba2']
        )
        fig_dist.update_layout(xaxis_title="Salary (USD)", yaxis_title="Count")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Key Statistics
    st.markdown("---")
    st.markdown("### 📊 Key Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("📊 Median Salary", f"${df['salary_in_usd'].median():,.0f}")
    with stat_col2:
        st.metric("💰 Max Salary", f"${df['salary_in_usd'].max():,.0f}")
    with stat_col3:
        st.metric("📉 Min Salary", f"${df['salary_in_usd'].min():,.0f}")
    with stat_col4:
        std_dev = df['salary_in_usd'].std()
        st.metric("📏 Std Deviation", f"${std_dev:,.0f}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Built with Streamlit • Powered by Machine Learning</p>
        <p style='font-size: 0.8rem;'>Model: {} | R² Score: {:.3f}</p>
    </div>
""".format(model_type, performance_metrics[model_type]["R² Score"]), unsafe_allow_html=True)
