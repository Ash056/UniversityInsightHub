import streamlit as st
import pandas as pd
import numpy as np
from auth import authenticate_user, check_permissions
from data_handler import DataHandler
from eda_module import EDAModule
from nlp_module import NLPModule
from topic_modeling import TopicModelingModule
from sentiment_analysis import SentimentAnalysisModule
from visualization import VisualizationModule
import os

# Configure page
st.set_page_config(
    page_title="University Survey Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # Authentication
    if not st.session_state.authenticated:
        st.title("🎓 University Survey Analytics Tool")
        st.markdown("### Please authenticate to access the system")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["Dean", "Administrator", "Analyst", "Viewer"])
            submit = st.form_submit_button("Login")
            
            if submit:
                if authenticate_user(username, password, role):
                    st.session_state.authenticated = True
                    st.session_state.user_role = role
                    st.session_state.username = username
                    st.success(f"Welcome, {username}! Logged in as {role}")
                    st.rerun()
                else:
                    st.error("Invalid credentials or insufficient permissions")
        return

    # Main application
    st.title("🎓 University Survey Analytics Dashboard")
    st.markdown(f"**Logged in as:** {st.session_state.username} ({st.session_state.user_role})")
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.session_state.data = None
        st.session_state.processed_data = None
        st.rerun()

    # Check permissions for data upload
    if check_permissions(st.session_state.user_role, "upload_data"):
        # File upload section
        st.sidebar.header("📁 Data Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Survey Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file containing survey responses"
        )
        
        if uploaded_file is not None:
            try:
                data_handler = DataHandler()
                df = data_handler.load_data(uploaded_file)
                st.session_state.data = df
                st.sidebar.success(f"Data loaded successfully! Shape: {df.shape}")
                
                # Data preview
                with st.sidebar.expander("Data Preview"):
                    st.dataframe(df.head())
                    
            except Exception as e:
                st.sidebar.error(f"Error loading data: {str(e)}")
    else:
        st.sidebar.warning("You don't have permission to upload data")

    # Main content tabs
    if st.session_state.data is not None:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Exploratory Data Analysis", 
            "🔤 Text Processing", 
            "🎯 Topic Modeling", 
            "😊 Sentiment Analysis", 
            "📈 Advanced Visualizations",
            "📋 Reports & Export"
        ])
        
        with tab1:
            if check_permissions(st.session_state.user_role, "view_eda"):
                eda_module = EDAModule()
                eda_module.run_analysis(st.session_state.data)
            else:
                st.warning("You don't have permission to view EDA")
        
        with tab2:
            if check_permissions(st.session_state.user_role, "view_nlp"):
                nlp_module = NLPModule()
                processed_data = nlp_module.run_preprocessing(st.session_state.data)
                st.session_state.processed_data = processed_data
            else:
                st.warning("You don't have permission to view NLP analysis")
        
        with tab3:
            if check_permissions(st.session_state.user_role, "view_topics"):
                if st.session_state.processed_data is not None:
                    topic_module = TopicModelingModule()
                    topic_module.run_topic_modeling(st.session_state.processed_data)
                else:
                    st.info("Please run text preprocessing first in the 'Text Processing' tab")
            else:
                st.warning("You don't have permission to view topic modeling")
        
        with tab4:
            if check_permissions(st.session_state.user_role, "view_sentiment"):
                sentiment_module = SentimentAnalysisModule()
                sentiment_module.run_sentiment_analysis(st.session_state.data)
            else:
                st.warning("You don't have permission to view sentiment analysis")
        
        with tab5:
            if check_permissions(st.session_state.user_role, "view_advanced"):
                viz_module = VisualizationModule()
                viz_module.create_advanced_visualizations(st.session_state.data)
            else:
                st.warning("You don't have permission to view advanced visualizations")
        
        with tab6:
            if check_permissions(st.session_state.user_role, "export_data"):
                st.header("📋 Reports & Export")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Generate Summary Report"):
                        report = generate_summary_report(st.session_state.data)
                        st.text_area("Summary Report", report, height=300)
                
                with col2:
                    if st.button("Export Data"):
                        csv = st.session_state.data.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="survey_analysis.csv",
                            mime="text/csv"
                        )
            else:
                st.warning("You don't have permission to export data")
    else:
        st.info("Please upload survey data to begin analysis")

def generate_summary_report(df):
    """Generate a comprehensive summary report"""
    report = f"""
UNIVERSITY SURVEY ANALYSIS REPORT
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW:
- Total Records: {len(df)}
- Total Columns: {len(df.columns)}
- Data Types: {df.dtypes.value_counts().to_dict()}

MISSING DATA:
{df.isnull().sum().to_string()}

NUMERICAL SUMMARY:
{df.describe().to_string()}

CATEGORICAL SUMMARY:
{df.select_dtypes(include=['object']).describe().to_string() if len(df.select_dtypes(include=['object']).columns) > 0 else 'No categorical columns found'}
    """
    return report

if __name__ == "__main__":
    main()
