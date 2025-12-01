import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import base64

# Page configuration
st.set_page_config(page_title="Cybersecurity Salary Explorer", page_icon="🕵️‍♂️", layout="wide")

# --- Banner Section ---
banner_path = Path("images/digital_rain_banner.jpg")
if banner_path.exists():
    with open(banner_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .banner {{
            width: 100%;
            height: 280px;
            background: url("data:image/jpeg;base64,{b64}") no-repeat center center;
            background-size: cover;
            border-radius: 12px;
            margin-bottom: 25px;
            position: relative;
        }}
        .banner-overlay {{
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.45);
            border-radius: 12px;
        }}
        .banner-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: white;
            font-size: 2.2rem;
            font-weight: 700;
            text-shadow: 0 2px 8px rgba(0,0,0,0.6);
        }}
        </style>
        <div class="banner">
            <div class="banner-overlay"></div>
            <div class="banner-text">🕵️‍♂️ Cybersecurity Jobs: Salary Explorer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("🕵️‍♂️ Cybersecurity Jobs: Salary Explorer")

# --- Team Members Section ---
st.subheader("👥 Team Members")
st.write("""
**Data Science Toolbox - Group Project**

- Aaron Tan Wen Zhuan (0137612)
- Hang Yih Wah (0138039)
- Katsutoki Ishii (0137257)
- Bryan Loo Zen Xuen (0137565)
""")

st.divider()

# --- Project Introduction ---
st.header("Welcome to the Cybersecurity Salary Explorer & Prediction Tool")

st.write("""
Explore the landscape of cybersecurity careers: **analyze salary trends, compare roles across countries, 
and predict your future earnings** based on real data.
""")

# --- Load Dataset ---
df = pd.read_csv('salaries_cyber_clean.csv')

# --- Dataset Overview ---
st.subheader("📊 Dataset at a Glance")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{df.shape[0]:,}")
with col2:
    st.metric("Unique Job Titles", df['job_title'].nunique())
with col3:
    st.metric("Years Covered", f"{df['work_year'].min()}–{df['work_year'].max()}")
with col4:
    st.metric("Countries", df['company_location'].nunique())

st.divider()

# --- Why Cybersecurity is Trending ---
st.subheader("🔥 Why Cybersecurity Jobs are Trending")

col1, col2 = st.columns(2)

with col1:
    st.write("""
    **🌐 Global surge in cyber threats and attacks**
    - Ransomware, data breaches, and cyberattacks are increasing
    - Organizations need more security professionals
    
    **💼 Remote work and digital transformation**
    - Remote work has expanded the attack surface
    - Companies need experts to secure distributed systems
    """)

with col2:
    st.write("""
    **💰 High salaries and job security**
    - Cybersecurity professionals earn competitive salaries
    - Strong demand ensures job stability
    
    **🎯 Skills gap—talent shortage creates opportunities**
    - Not enough qualified professionals to fill positions
    - Great opportunities for career growth
    """)

st.divider()

# --- Importance of This Tool ---
st.subheader("🎯 Why Use This Tool?")

col1, col2 = st.columns(2)

with col1:
    st.write("""
    **👨‍💼 Job Seekers:** Know your worth before negotiating offers
    
    **🎓 Students:** Make data-driven career decisions
    
    **🏢 Employers:** Benchmark salaries to attract/retain talent
    """)

with col2:
    st.write("""
    **🔬 Researchers:** Analyze workforce and compensation trends
    
    **👨‍🏫 Educators:** Advise students with up-to-date industry info
    """)

st.divider()

# --- Visualizations ---
st.subheader("📈 Key Insights")

# Line Chart: Salary Trend Over Years
st.markdown("**Salary Trend Over Years**")
st.write("This chart shows how the average salary in cybersecurity has changed over the years. "
         "An upward trend indicates growing demand and value for cybersecurity professionals.")
yearly_avg = df.groupby('work_year')['salary_in_usd'].mean().reset_index()
fig1 = px.line(yearly_avg, x='work_year', y='salary_in_usd', markers=True,
               labels={'work_year': 'Year', 'salary_in_usd': 'Average Salary (USD)'})
st.plotly_chart(fig1, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    # Bar Chart: Top 5 Job Titles with Different Colors
    st.markdown("**Top 5 Most Common Job Titles**")
    st.write("These are the most frequently occurring job positions in the cybersecurity field, "
             "showing where the highest demand exists.")
    top_jobs = df['job_title'].value_counts().head(5).reset_index()
    top_jobs.columns = ['job_title', 'count']
    fig2 = px.bar(top_jobs, x='job_title', y='count',
                  labels={'job_title': 'Job Title', 'count': 'Count'},
                  color='job_title',
                  color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    # Pie Chart: Experience Level Distribution
    st.markdown("**Experience Level Distribution**")
    st.write("This shows the breakdown of positions by experience level, "
             "helping you understand which career stage has the most opportunities.")
    exp_dist = df['experience_level'].value_counts().reset_index()
    exp_dist.columns = ['experience_level', 'count']
    fig3 = px.pie(exp_dist, values='count', names='experience_level',
                  color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --- Sample Data ---
st.subheader("🔍 Sample Data")
st.dataframe(df.head(10))

# --- Column Descriptions ---
st.subheader("📋 Column Descriptions")
column_info = {
    "work_year": "Year of the salary record",
    "experience_level": "Experience level of employee",
    "employment_type": "Full-time, part-time, etc.",
    "job_title": "Role or job title",
    "salary": "Salary in original currency",
    "salary_currency": "Currency of salary",
    "salary_in_usd": "Salary standardized to USD",
    "employee_residence": "Country of employee",
    "remote_ratio": "Remote work percentage (0–100)",
    "company_location": "Location of company HQ",
    "company_size": "Company size (S/M/L)"
}
st.table(pd.DataFrame(list(column_info.items()), columns=["Column", "Description"]))