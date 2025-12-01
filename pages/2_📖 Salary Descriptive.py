import streamlit as st
import pandas as pd
import plotly.express as px
import pycountry
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json

# ------------- PAGE CONFIG -------------
st.set_page_config(page_title="Salary Descriptive in Cybersecurity Workforce", page_icon="📈", layout="wide")


# ----------- Title & Intro -----------
st.title("📒 Salary Descriptive: Cybersecurity Job Market Insights")

st.markdown("""
Explore salary patterns, job distribution, and structural insights in the cybersecurity industry.
""")

# Load your first Lottie animation
with open("images/Data Extraction.json", "r") as f:
    lottie_1 = json.load(f)

# Load your second Lottie animation
with open("images/illustration graph.json", "r") as f:
    lottie_2 = json.load(f)

# Create 3 columns: empty | animation 1 | animation 2 | empty
col_left, col_mid1, col_mid2, col_right = st.columns([1, 2, 2, 1])

with col_mid1:
    st_lottie(lottie_1, speed=1, loop=True, width=250, height=250, key="lottie1")

with col_mid2:
    st_lottie(lottie_2, speed=1, loop=True, width=250, height=250, key="lottie2")

# ----------- Load Data -----------
df = pd.read_csv('salaries_cyber_clean.csv')

# ----------- MAPPING LABELS -----------
employment_map = {
    'FT': 'FT (Full Time)', 'PT': 'PT (Part Time)',
    'CT': 'CT (Contract)', 'FL': 'FL (Freelance)'
}
experience_map = {
    'EN': 'EN (Entry)', 'MI': 'MI (Mid)',
    'SE': 'SE (Senior)', 'EX': 'EX (Executive)'
}
df['employment_type_full'] = df['employment_type'].map(employment_map)
df['experience_level_full'] = df['experience_level'].map(experience_map)

def remote_mode(ratio):
    if ratio == 0: return "Onsite"
    elif ratio == 100: return "Remote"
    else: return "Hybrid"
df['remote_mode'] = df['remote_ratio'].apply(remote_mode)

size_map = {"S": "Small", "M": "Medium", "L": "Large"}
exp_map = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Exec"}
df['Company Size'] = df['company_size'].map(size_map)
df['Experience'] = df['experience_level'].map(exp_map)

# ----------- TABS FOR NAVIGATION -----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Highest Average Salary Profession",
    "Job Market Saturation",
    "Salary Heatmap: Comp Size × Exp Level",
    "Average Salary: Exp Level × Emp Type × Remote Stat",
    "Job Distribution Treemap"
])

# ----------- TAB 1: Highest Average Salary Profession -----------
with tab1:
    st.markdown("""
    <h4>📈 <i>Highest Average Salary by Profession (Top 15)</i></h4>
    <ul>
        <li>Horizontal bar chart showing the 15 job titles with the highest average annual salary.</li>
        <li>Bar shade intensity represents the salary level.</li>
    </ul>
    """, unsafe_allow_html=True)
    
    avg_salary_job = df.groupby('job_title', as_index=False)['salary_in_usd'].mean()
    avg_salary_job = avg_salary_job.sort_values('salary_in_usd', ascending=False).head(15)
    avg_salary_job['salary_label'] = avg_salary_job['salary_in_usd'].apply(lambda x: f"${int(x/1000)}k")
    fig_barh = px.bar(
        avg_salary_job,
        y='job_title',
        x='salary_in_usd',
        orientation='h',
        color='salary_in_usd',
        color_continuous_scale='teal',
        labels={'job_title': 'Job Title', 'salary_in_usd': 'Average Salary (USD)'},
        text='salary_label'
    )
    fig_barh.update_traces(
        textposition='auto', textfont_size=14,
        marker_line_color='#39FF14', marker_line_width=0.2
    )
    fig_barh.update_layout(
        yaxis=dict(categoryorder='total ascending', tickfont=dict(size=13)),
        xaxis=dict(title="Average Salary (USD)", tickfont=dict(size=13)),
        margin=dict(l=180, r=30, t=60, b=40), height=650,
        coloraxis_colorbar=dict(title="Average Salary (USD)")
    )
    st.plotly_chart(fig_barh, use_container_width=True)

    st.markdown("""
    <span style='color: #888; font-size: 1.05em'>
    As seen above, leadership and tech-lead roles dominate the top-paying cybersecurity positions. The range spans from $163k to $375k, with most top jobs falling within $175k-$215k (e.g., Data Infrastructure Lead to Data Lead). The Data Science Tech Lead stands out with the highest average, reflecting both its leadership and specialized expertise premium.
    </span>
    """, unsafe_allow_html=True)

    # adding margin div for spacing
    st.markdown("""
        <div style="margin-top: 40px; margin-bottom: 30px;">
    """, unsafe_allow_html=True)

    # --------- Searchable Table for Top 15 Jobs ---------
    st.markdown("""
    <h4>🔎 Explore the Top 15 Highest Paying Cybersecurity Roles</h4>
    <p>Type a job title (or part of one) to quickly find details!</p>
    """, unsafe_allow_html=True)

    search = st.text_input("Search job title in Top 15...", value="", key="top15_search")
    filtered = avg_salary_job[avg_salary_job['job_title'].str.contains(search, case=False, na=False)]

    # Format for display
    df_table = filtered.copy()
    df_table['salary_in_usd'] = df_table['salary_in_usd'].map('${:,.0f}'.format)
    df_table = df_table.rename(columns={
        'job_title': 'Job Title',
        'salary_in_usd': 'Average Salary',
        'salary_label': 'Salary Label'
    })
    # Only show relevant columns
    st.dataframe(
        df_table[['Job Title', 'Average Salary']],
        hide_index=True,
        use_container_width=True
    )



# ----------- TAB 2: Job Market Saturation (Violin Plot) -----------
with tab2:
    st.markdown("""
    <h4>🎻 <i>Salary Distribution by Top 10 Roles</i></h4>
    <ul>
        <li>Violin plot shows the salary spread and outliers for the 10 most common job titles.</li>
        <li>Thick center = common salaries; thin tails = rare/outlier salaries.</li>
    </ul>
    """, unsafe_allow_html=True)
    
    top_titles = df['job_title'].value_counts().nlargest(10).index
    df_top_jobs = df[df['job_title'].isin(top_titles)]
    fig_violin_job = px.violin(
        df_top_jobs,
        x="job_title", y="salary_in_usd", color="job_title",
        box=True, points="outliers", color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_violin_job.update_layout(
        xaxis_title="Job Title", yaxis_title="Salary (USD)", showlegend=False
    )
    fig_violin_job.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_violin_job, use_container_width=True)

    st.markdown("""
    <span style='color: #888; font-size: 1.05em'>
    Most cybersecurity job titles show a wide range of salaries, with “tails” indicating the occasional very high or low outlier. Senior and specialist roles (e.g., Manager, Tech Lead) tend to have broader distributions, reflecting more pay variation. In contrast, common roles like “Security Engineer” are tightly centered, indicating stable market rates.
    </span>
    """, unsafe_allow_html=True)


# ----------- TAB 3: Salary Heatmap by Comp Size & Exp Level -----------
with tab3:
    st.markdown("""
    <h4>🌡️ <i>Salary Heatmap: Company Size × Experience Level</i></h4>
    <ul>
        <li>This heatmap shows the <b>average salary</b> by company size (Small, Medium, Large) and experience level (Entry, Mid, Senior, Exec).</li>
        <li>Darker shades = higher average salary.</li>
    </ul>
    """, unsafe_allow_html=True)
    
    # Create average salary pivot
    heatmap_data = df.pivot_table(
        index='Company Size',
        columns='Experience',
        values='salary_in_usd',
        aggfunc='mean'
    )
    heatmap_data = heatmap_data.reindex(index=["Small", "Medium", "Large"])
    heatmap_data = heatmap_data[["Entry", "Mid", "Senior", "Exec"]]
    
    fig_heatmap = px.imshow(
        heatmap_data,
        text_auto=True,
        color_continuous_scale='viridis',
        aspect='auto',
        labels=dict(x="Experience Level", y="Company Size", color="Avg Salary (USD)"),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("""
    <span style='color: #888; font-size: 1.05em'>
    <b>Insight:</b> Salaries predictably rise with both experience and company size. Large companies offer the highest pay at every level. Notably, executive-level salaries at large firms are dramatically higher than at small/medium companies. Entry-level pay is similar regardless of size, but the “salary gap” widens significantly with seniority.
    </span>
    """, unsafe_allow_html=True)


# ----------- TAB 4: Interactive Salary Comparison (Choose Bar Chart Type) -----------

with tab4:
    st.markdown("""
    <h4>📊 <i>Custom Salary Comparison</i></h4>
    <ul>
        <li>Select which dimension to compare: <b>Remote Type</b>, <b>Experience Level</b>, or <b>Employment Type</b>.</li>
        <li>The bar chart will display average salary for each category.</li>
    </ul>
    """, unsafe_allow_html=True)
    
    chart_type = st.selectbox(
        "Compare Average Salary by...",
        ["Remote Type", "Experience Level", "Employment Type"],
        index=0
    )

    if chart_type == "Remote Type":
        plot_data = df.groupby('remote_mode')['salary_in_usd'].mean().reset_index()
        fig = px.bar(
            plot_data, x='remote_mode', y='salary_in_usd',
            text='salary_in_usd',
            color='remote_mode',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={'remote_mode': "Remote Type", 'salary_in_usd': "Average Salary (USD)"}
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Average Salary (USD)", xaxis_title=None, height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <span style='color: #888; font-size: 1.05em'>
        <b>Insight:</b> Remote and hybrid roles tend to offer higher average salaries compared to onsite roles, reflecting the global demand and flexibility for remote cybersecurity professionals.
        </span>
        """, unsafe_allow_html=True)

    elif chart_type == "Experience Level":
        plot_data = df.groupby('experience_level_full')['salary_in_usd'].mean().reset_index()
        fig = px.bar(
            plot_data, x='experience_level_full', y='salary_in_usd',
            text='salary_in_usd',
            color='experience_level_full',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={'experience_level_full': "Experience Level", 'salary_in_usd': "Average Salary (USD)"}
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Average Salary (USD)", xaxis_title=None, height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <span style='color: #888; font-size: 1.05em'>
        <b>Insight:</b> Salary increases significantly with experience level. Executive and senior roles command the highest pay, while entry-level positions start much lower.
        </span>
        """, unsafe_allow_html=True)

    else:  # Employment Type
        plot_data = df.groupby('employment_type_full')['salary_in_usd'].mean().reset_index()
        fig = px.bar(
            plot_data, x='employment_type_full', y='salary_in_usd',
            text='salary_in_usd',
            color='employment_type_full',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={'employment_type_full': "Employment Type", 'salary_in_usd': "Average Salary (USD)"}
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Average Salary (USD)", xaxis_title=None, height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <span style='color: #888; font-size: 1.05em'>
        <b>Insight:</b> Full-time jobs offer the highest average salary, followed by contract and freelance roles. Part-time work pays significantly less, as expected in the cybersecurity sector.
        </span>
        """, unsafe_allow_html=True)


# ----------- TAB 5: Job Distribution Treemap -----------
with tab5:

    st.markdown("""
    <h4>🗂️ <i>Cybersecurity Role Treemap: Top 25 Most Common Jobs</i></h4>
    <ul>
        <li>Treemap shows the top 25 most frequent cybersecurity job titles.</li>
        <li>Box size = number of job records (count).</li>
        <li>Hover to see job title, median salary, average salary, and total count.</li>
    </ul>
    """, unsafe_allow_html=True)

    # ---- Prepare Top 25 Only ----
    job_counts = (
        df.dropna(subset=['salary_in_usd'])
        .groupby('job_title')
        .agg(
            avg_salary=('salary_in_usd', 'mean'),
            median_salary=('salary_in_usd', 'median'),
            count=('salary_in_usd', 'count')
        )
        .sort_values(by='count', ascending=False)
        .head(25)                      # ← LIMIT TO TOP 25
        .reset_index()
    )

    # ---- TREEMAP ----
    palette = px.colors.qualitative.Vivid + px.colors.qualitative.Pastel

    fig_tree = px.treemap(
        job_counts,
        path=['job_title'],
        values='count',
        color='count',                      # simple clean color scale
        color_continuous_scale='Tealgrn',
        custom_data=['job_title', 'avg_salary', 'median_salary', 'count']
    )

    fig_tree.update_traces(
        texttemplate="<b>%{label}</b><br>n=%{customdata[3]:,}",     # SHOW COUNT ONLY
        hovertemplate=(
            "<b>%{customdata[0]}</b><br><br>"
            "Avg Salary: $%{customdata[1]:,.0f}<br>"
            "Median Salary: $%{customdata[2]:,.0f}<br>"
            "Count: %{customdata[3]:,}<extra></extra>"
        )
    )

    fig_tree.update_layout(
        height=600,
        margin=dict(l=40, r=40, t=40, b=40)     # ← CLEAN MARGINS
    )

    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("""
        <span style='color: #bbbbbb; font-size: 1.05em'>
        These top 25 most common cybersecurity roles give a quick snapshot of market demand. Larger tiles represent
        more frequently occurring job titles. Higher-paying specialist roles may appear smaller, while common roles  
        like Security Engineer take up more space due to job volume.
        </span>
    """, unsafe_allow_html=True)

    # ==== Close margin div ====
    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # ----------------- SEARCHABLE JOB TABLE ---------------------
    # ============================================================

    # adding margin div for spacing
    st.markdown("""
        <div style="margin-top: 40px; margin-bottom: 30px;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <h4>🔍 Explore All Cybersecurity Jobs & Salaries</h4>
    <p>Use the search bar to instantly find salary statistics for any cybersecurity role across the dataset.</p>
    """, unsafe_allow_html=True)

    search = st.text_input("Search by job title...", value="", key="job_search")

    # Table uses FULL dataset, not only top 25
    full_jobs = (
        df.dropna(subset=['salary_in_usd'])
        .groupby('job_title')
        .agg(
            avg_salary=('salary_in_usd', 'mean'),
            median_salary=('salary_in_usd', 'median'),
            count=('salary_in_usd', 'count')
        )
        .reset_index()
    )

    filtered = full_jobs[full_jobs['job_title'].str.contains(search, case=False, na=False)]

    # Format for table
    df_table = filtered.copy()
    df_table['avg_salary'] = df_table['avg_salary'].map('${:,.0f}'.format)
    df_table['median_salary'] = df_table['median_salary'].map('${:,.0f}'.format)

    df_table = df_table.rename(columns={
        'job_title': 'Job Title',
        'avg_salary': 'Avg Salary',
        'median_salary': 'Median Salary',
        'count': 'Records'
    })

    st.dataframe(df_table.reset_index(drop=True), hide_index=True, use_container_width=True)
