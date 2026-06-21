import streamlit as st
import pandas as pd
from model import build_and_train_pipeline

st.set_page_config(page_title="AI Impact on Students Predictor", layout="centered")

st.title("🎓 AI Impact on Students Performance Predictor")
st.write(
    "This app trains a live machine learning model using the **Kaggle Impact of AI on Students Dataset** "
    "and dynamically runs predictions for estimated **Post-Semester GPAs** based on custom inputs."
)

@st.cache_resource
def get_cached_trained_pipeline():
    try:
        pipeline, X_raw = build_and_train_pipeline()
        return pipeline, X_raw
    except Exception as e:
        st.error(f"Failed to load or process dataset from Kaggle: {e}")
        return None, None

with st.spinner("Downloading dataset from Kaggle and training model... Please wait..."):
    pipeline, X_raw = get_cached_trained_pipeline()

if pipeline is not None and X_raw is not None:
    st.success("✅ Dataset pulled and Model Pipeline successfully loaded!")
    
    st.header("📋 Input Student Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        major = st.selectbox("Major Category", options=list(X_raw['Major_Category'].unique()))
        year = st.selectbox("Year of Study", options=list(X_raw['Year_of_Study'].unique()))
        use_case = st.selectbox("Primary GenAI Use Case", options=list(X_raw['Primary_Use_Case'].unique()))
        skill = st.selectbox("Prompt Engineering Skill", options=list(X_raw['Prompt_Engineering_Skill'].unique()))
        paid_sub = st.checkbox("Paid AI Subscription?", value=False)

    with col2:
        pre_gpa = st.slider("Pre-Semester GPA", min_value=1.0, max_value=4.0, value=3.15, step=0.01)
        ai_hours = st.slider("Weekly GenAI Usage Hours", min_value=0.0, max_value=40.0, value=8.5, step=0.5)
        study_hours = st.slider("Traditional Study Hours/Week", min_value=1.0, max_value=36.0, value=11.0, step=0.5)
        tool_div = st.slider("Tool Diversity (Count of unique AI Tools)", min_value=1, max_value=5, value=2)
        ai_dep = st.slider("Perceived AI Dependency Scale", min_value=1, max_value=10, value=4)
        anxiety = st.slider("Anxiety Level During Exams", min_value=1, max_value=10, value=4)

    # 1. Structure values to match the exact keys defined in model features
    user_inputs = {
        'Major_Category': major,
        'Year_of_Study': year,
        'Pre_Semester_GPA': pre_gpa,
        'Weekly_GenAI_Hours': ai_hours,
        'Primary_Use_Case': use_case,
        'Prompt_Engineering_Skill': skill,
        'Tool_Diversity': tool_div,
        'Paid_Subscription': paid_sub,
        'Traditional_Study_Hours': study_hours,
        'Perceived_AI_Dependency': ai_dep,
        'Anxiety_Level_During_Exams': anxiety
    }

    # 2. Build tracking dataframe
    input_data = pd.DataFrame([user_inputs])

    # 3. Align column order exactly with X_raw format
    input_data = input_data[X_raw.columns]

    st.write("---")
    
    if st.button("🔮 Predict Post-Semester GPA", type="primary"):
        prediction = pipeline.predict(input_data)[0]
        prediction = max(1.0, min(4.0, prediction)) # Keep inside realistic boundaries
        
        st.metric(label="Predicted Post-Semester GPA Outcome", value=f"{prediction:.2f}")
        
        gpa_diff = prediction - pre_gpa
        if gpa_diff >= 0:
            st.success(f"📈 Projected GPA improvement of **+{gpa_diff:.2f}** grade points.")
        else:
            st.warning(f"📉 Projected GPA decline of **{gpa_diff:.2f}** grade points.")
else:
    st.error("Application configuration failed. Ensure your Streamlit Cloud environment has adequate internet access.")