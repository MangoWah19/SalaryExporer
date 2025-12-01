import streamlit as st
import base64
from pathlib import Path
import os 

# --- Page Configuration ---
st.set_page_config(page_title="About Us", page_icon="👥")

# --- Function to encode image to Base64 ---
def get_image_as_base64(path):
    """Loads the image at the specified path as a Base64 string."""
    if not Path(path).exists():
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None

# --- Project Root Directory Setup ---
# Gets the directory of the script (1_👥 About Us.py) and resolves its parent (Assignment folder).
ROOT_DIR = Path(__file__).parent.parent.resolve()

# --- Page Title ---
st.title("👥 About Us")
st.write("---")

# --- Section 1: Who We Are ---
# --- Display University Logo and Team Introduction in two columns ---
col1, col2 = st.columns([1, 3])  # Ratio: 1 for Logo, 3 for Text

with col1:
    st.subheader("Who We Are")
    # Build the absolute path to the university logo
    uni_logo_filename = "UOWM_Logo.png"
    uni_logo_path = Path("images/UOWM_Logo.png")
    
    if uni_logo_path.exists():
        # st.image handles Pathlib objects correctly
        st.image(str(uni_logo_path), width=150)
    else:
        st.warning(f"University Logo image ({uni_logo_filename}) not found. Please ensure it is in the Assignment folder.")

with col2:
    st.subheader("")
    st.write("""
    We are a dedicated student team from the **Faculty of Computer Science (Honours)** at **UOW Malaysia**, driven by a passion for Data Science and technology.
    
    This project was created as part of the assessment for the 'Data Science Toolbox (XBDS2034/N)' course. Our goal is to apply the knowledge we've gained in a practical manner to develop a valuable and insightful tool.
    """)

st.write("---")

# --- Section 2: Our Mission & Objective ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Our Mission")
    st.info(
        "**To support career development by providing practical salary insights to cybersecurity professionals and students.**"
    )

with col2:
    st.subheader("Our Objectives")
    st.markdown("""
    - **Visualization**: To visualize global cybersecurity job salary trends using interactive charts.
    - **Prediction**: To offer a Machine Learning model that predicts salaries based on user-inputted conditions (e.g., experience, job title).
    - **Analysis**: To analyze and share insights into the impact of job role, experience level, location, and remote work on compensation.
    """)

st.write("---")

# --- Section 3: Our Team ---
st.header("Meet the Team")

# --- Define Team Member Information (UPDATED to ensure filename consistency) ---
team_members = [
    {
        "name": "Aaron",
        "role": "UI/UX & Homepage Lead",
        "image_filename": "Aaron-profile.png",  
        "description": "Led the design and structure of the dashboard, implementing the main navigation and visual components found in `Homepage.py`.",
        "github": "" # Empty GitHub link to prevent placeholder display
    },
    {
        "name": "Bryan",
        "role": "Predictive Model Engineer",
        "image_filename": "Bryan-profile.png",  
        "description": "Developed and implemented the Machine Learning model in `3_🤖 Predictive Model.py`, focusing on model selection, training, and performance evaluation.",
        "github": "" 
    },
    {
        "name": " Yih Wah",
        "role": "Exploratory Data Analyst (EDA)",
        "image_filename": "YihWah-profile.png",  
        "description": "Responsible for all data preprocessing, cleaning, and creating the data visualizations and key insights presented in `2_📈 Salary Descriptive.py`.",
        "github": "" 
    },
    {
        "name": " Katsutoki",
        "role": "Documentation & About Us Lead",
        "image_filename": "Katsutoki-profile.png",  
        "description": "Managed team documentation, created the `1_👥 About Us.py` page, and contributed to the final report's writing and proofreading.",
        "github": "" 
    }
]

# --- Display members in columns (Automatically adjusts to team size) ---
cols = st.columns(len(team_members))

for i, member in enumerate(team_members):
    with cols[i]:
        avatar_path  = Path(f"images/{member['image_filename']}")
        
        # Check existence and get Base64 string
        b64_image = get_image_as_base64(str(avatar_path)) 
        
        if b64_image:
            st.markdown(
                f"""
                <div style="width:100%; text-align:center;">
                    <a href="{member['github']}" target="_blank"
                    title="View GitHub for {member['name']}"
                    style="text-decoration:none; color:inherit; display:inline-block;">
                        <img src="data:image/png;base64,{b64_image}" alt="{member['name']}"
                            style="width:120px; height:120px; border-radius:50%;
                                    object-fit:cover; border:2px solid #4CAF50;">
                        <div style="margin-top:10px;">
                            <p style="margin:0; font-size:1.05em; font-weight:600; text-align:center;">
                                {member['name']}
                            </p>
                            <p style="margin:0; font-size:0.9em; color:#666; text-align:center;">
                                {member['role']}
                            </p>
                        </div>  
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                f"Avatar image ({member['image_filename']}) not found or failed to encode."
            )
            st.markdown(
                f"""
                <div style="width:100%; text-align:center;">
                    <div style="width:120px; height:120px; border-radius:50%;
                                background-color:#eee; display:inline-block;
                                line-height:120px; color:#888; font-size:0.8em;
                                border:2px solid #ddd;">
                        Avatar N/A
                    </div>
                    <div style="margin-top:8px;">
                        <p style="margin:0; font-size:1.05em; font-weight:600; text-align:center;">
                            {member['name']}
                        </p>
                        <p style="margin:0; font-size:0.9em; color:#666; text-align:center;">
                            {member['role']}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.write("---")


# --- Quote about the Future ---
st.markdown(
    """
    > "The future of cybersecurity is not just about building taller walls, but about building smarter bridges between people, processes, and technology."
    """,
    unsafe_allow_html=True
)

st.write("---")

# --- Footer: Submission Details ---
st.markdown(
    """
    <div style="text-align: center; color: grey; font-size: 0.9em; padding-top: 20px;">
        <p><strong>Submission Details</strong></p>
        <p>Course: Data Science Toolbox (XBDS2034/N)</p>
        <p>Lecturer: Norsyela Muhammad Noor Mathivanan</p>
        <p>Due Date: 01 December 2025</p>
    
    </div>
    """,
    unsafe_allow_html=True
)