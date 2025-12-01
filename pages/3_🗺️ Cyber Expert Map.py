import streamlit as st
import pandas as pd
import plotly.express as px
import pycountry

# ----------- Page Config -----------
st.set_page_config(
    page_title="Cybersecurity Salary & Workforce (Globe)",
    page_icon="🗺️",
    layout="wide"
)

# ----------- Minimal CSS for spacing -----------
st.markdown("""
<style>
.main .block-container {padding: 1.5rem 10rem 2rem 10rem;}
[data-testid="stSidebar"] {min-width: 200px;}
div[data-testid="stRadio"] > label { margin-top: 0.6rem !important; }
div[data-testid="stRadio"] { margin-bottom: 1.2rem !important; }
label[data-testid="stMarkdownContainer"] p { margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ----------- Title ----------- 
st.title("🌍 Global Cybersecurity Workforce & Salary Distribution")
st.markdown("""
Explore how cybersecurity talent and salaries are spread across the world.  
This interactive 3D globe visualizes **employee residence**, **company location**,  
and **regional salary differences**.  
""")

# ----------- Load Data -----------
df = pd.read_csv("salaries_cyber_clean.csv")

# ----------- Metric Selector -----------
metric = st.radio(
    "**Metric**",
    ["Average Salary by Company Location", "Number of Employees by Residence"],
    horizontal=True
)

# ----------- Region Selector -----------
region = st.radio(
    "**Map Scope Selection**",
    ["NA", "EU", "SA", "AF", "AS"],
    captions=["North America", "Europe", "South America", "Africa", "Asia"],
    horizontal=True
)

# ----------- Continent Mapping -----------
CONTINENT_MAP = {
    "NA": {"US","CA","MX"},
    "EU": {"GB","FR","DE","ES","IT","NL","PL","SE","NO","FI","BE","DK","PT","IE","CH","AT","CZ","RO","HU","GR","BG","SK","HR","SI","EE","LV","LT","LU","IS"},
    "SA": {"BR","AR","CL","CO","PE","VE","UY","PY","BO","EC","GY","SR"},
    "AF": {"ZA","NG","EG","DZ","MA","KE","TZ","UG","GH","CM","CI","ET","SN","ZM","ZW","SD","RW","TN","MW","NA"},
    "AS": {"CN","JP","KR","IN","ID","MY","SG","TH","PH","VN","BD","PK","LK","AE","SA","IL","IR","IQ","TR","KZ","KG","MM","QA","KW"}
}
region_set = CONTINENT_MAP.get(region, set())

REGION_CENTER = {
    "NA": {"lon": -100, "lat": 40},
    "EU": {"lon": 10, "lat": 50},
    "SA": {"lon": -60, "lat": -15},
    "AF": {"lon": 20, "lat": 5},
    "AS": {"lon": 95, "lat": 30}
}
center = REGION_CENTER[region]

# ----------- Build dataframe for map -----------
if metric == "Average Salary by Company Location":
    map_df = df.groupby("company_location", as_index=False)["salary_in_usd"].mean()
    map_df = map_df.rename(columns={
        "company_location": "Country",
        "salary_in_usd": "Average Salary (USD)"
    })
    color_col = "Average Salary (USD)"
    hover_data = {"Average Salary (USD)": True, "Country_Code": True}
    color_scale = px.colors.sequential.Viridis
else:
    map_df = df["employee_residence"].value_counts().reset_index()
    map_df.columns = ["Country", "Number of Employees"]
    color_col = "Number of Employees"
    hover_data = {"Number of Employees": True, "Country_Code": True}
    color_scale = px.colors.sequential.Plasma

# ----------- Convert Alpha-2 to Alpha-3 + names -----------
def a2_to_a3(code):
    try: return pycountry.countries.get(alpha_2=code).alpha_3
    except: return None

def a2_to_name(code):
    try: return pycountry.countries.get(alpha_2=code).name
    except: return code

map_df["Country_Code"] = map_df["Country"].apply(a2_to_a3)
map_df["Country_Name"] = map_df["Country"].apply(a2_to_name)
map_df = map_df.dropna(subset=["Country_Code"])

# Apply region filter
if region_set:
    map_df = map_df[map_df["Country"].isin(region_set)]


BORDER = "#0A4F3B"
LAND = "#2E2E2E"
OCEAN = "#0096C7"
BG = "rgba(0,0,0,0)"

# ----------- Build Globe -----------
fig = px.choropleth(
    map_df,
    locations="Country_Code",
    locationmode="ISO-3",
    color=color_col,
    hover_name="Country_Name",
    hover_data=hover_data,
    color_continuous_scale=color_scale
)

fig.update_geos(
    projection=dict(
        type="orthographic",
        rotation=dict(lon=center["lon"], lat=center["lat"])
    ),
    showcountries=True,
    showcoastlines=True,
    showland=True,
    showocean=True,
    landcolor=LAND,
    oceancolor=OCEAN,
    bgcolor=BG,
    countrycolor=BORDER,
    countrywidth=0.4,
    coastlinecolor=BORDER,
    coastlinewidth=0.4
)

fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    geo_bgcolor=BG,
    margin=dict(r=0, t=50, l=0, b=0)
)



st.plotly_chart(fig, use_container_width=True)

# ----------- Country Selection -----------
st.markdown("### Country Selection")
country_name = st.selectbox("", sorted(map_df["Country_Name"].unique()))
row = map_df[map_df["Country_Name"] == country_name].iloc[0]
country_a2 = row["Country"]

# Filter original dataframe
if metric == "Average Salary by Company Location":
    sub = df[df["company_location"] == country_a2]
else:
    sub = df[df["employee_residence"] == country_a2]

total_records = len(sub)

# Safe mode helper
def safe_mode(series):
    return series.mode()[0] if not series.mode().empty else "N/A"

# Build summary table
if metric == "Average Salary by Company Location":
    summary_df = pd.DataFrame({
        "Metric": [
            "Total records",
            "Average salary (USD)",
            "Median salary (USD)",
            "Highest-paying role",
            "Most common job title",
            "Most common experience level"
        ],
        "Value": [
            f"{total_records:,}",
            f"${sub['salary_in_usd'].mean():,.2f}" if total_records else "N/A",
            f"${sub['salary_in_usd'].median():,.2f}" if total_records else "N/A",
            sub.loc[sub["salary_in_usd"].idxmax(), "job_title"] if total_records else "N/A",
            safe_mode(sub["job_title"]),
            safe_mode(sub["experience_level"])
        ]
    })
else:
    summary_df = pd.DataFrame({
        "Metric": [
            "Total records (residence)",
            "Average salary (USD)",
            "Most common job title",
            "Most common experience level"
        ],
        "Value": [
            f"{total_records:,}",
            f"${sub['salary_in_usd'].mean():,.2f}" if total_records else "N/A",
            safe_mode(sub["job_title"]),
            safe_mode(sub["experience_level"])
        ]
    })

st.markdown(f"#### Overview for **{country_name}**")
st.dataframe(summary_df, use_container_width=True)

# ----------- Disclaimer -----------
st.write("""
**Note:**  
This dataset is based on publicly available and user-submitted data.  
Salary and job distributions may not fully represent real-world global conditions.
""")
