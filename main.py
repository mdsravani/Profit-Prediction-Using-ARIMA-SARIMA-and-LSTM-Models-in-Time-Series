import streamlit as st
import pandas as pd
import numpy as np
from src.visualizations import DataVisualizer
from src.styles import TITLE_STYLE, SIDEBAR_STYLE
from src.streamlit_utils import DataContent, DataTable
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
import streamlit.components.v1 as components
import io
import os
import importlib.util

st.set_page_config(
    page_title="Profit Prediction",
    layout="wide",  # Expands content area
    initial_sidebar_state="expanded",  # Keeps sidebar open
)

def convert_df_to_csv(df):
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()

def main():
    st.markdown(TITLE_STYLE, unsafe_allow_html=True)
    st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

    st.markdown('<h1 class="styled-title">PROFIT FORECASTING USING TIME SERIES FORECASTING TECHNIQUES</h1>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-title">Select Options</div>', unsafe_allow_html=True)
    

    if 'page' not in st.session_state:
        st.session_state['page'] = "Problem Statement"

    if "df" not in st.session_state:
        st.session_state.df = None 
    
    if 'pre_df' not in st.session_state:
        st.session_state.pre_df = None
    
    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = None

    # Sidebar buttons
    if st.sidebar.button("Problem Statement"):
        st.session_state['page'] = "Problem Statement"

    if st.sidebar.button("Project Data Description"):
        st.session_state['page'] = "Project Data Description"

    if st.sidebar.button("Sample Training Data"):
        st.session_state['page'] = "Sample Training Data"

    if st.sidebar.button("Know About Data"):
        st.session_state['page'] = "Know About Data"

    if st.sidebar.button("Data Preprocessing"):
        st.session_state['page'] = "Data Preprocessing"

    if st.sidebar.button("Exploratory Data Analysis"):
        st.session_state['page'] = "Exploratory Data Analysis"

    if st.sidebar.button("Machine Learning Models Used"):
        st.session_state['page'] = "Machine Learning Models Used"

    if st.sidebar.button("Model Predictions"):
        st.session_state['page'] = "Model Predictions"

################################################################################################################

    if st.session_state['page']== "Problem Statement":
        st.image(r"C:\Profit prediction\revenue_forecasting_1696220839.jpg", width=300)
        st.markdown(DataContent.problem_statement)
    
    elif  st.session_state['page'] == "Project Data Description":
        st.markdown(DataContent.project_data_details)

    elif st.session_state['page'] == "Sample Training Data":
        st.markdown("## 📊 Training Data Preview")
        st.write("🔍 Below is an *interactive table* displaying the first 100 rows:")
        file_path = r"C:\Profit prediction\data\dataset\1M_data.csv"
        st.session_state.df = pd.read_csv(file_path)
        data_table = DataTable(df=st.session_state.df)
        data_table.display_table()


    elif  st.session_state['page'] == "Know About Data":
        file_path = r"C:\Profit prediction\data\dataset\1M_data.csv"
        st.session_state.df = pd.read_csv(file_path)
        st.header("Data Information")

        if "profile_report_generated" not in st.session_state:
            with st.status("⏳ Generating Overall Data Profile Report...", expanded=True) as status:
                profile = ProfileReport(st.session_state.df, explorative=True)
                profile.to_file("ydata_profiling_report.html")
                st.session_state["profile_report_generated"] = True  # Mark as generated
                status.update(label="✅ Report Generated Successfully!", state="complete")

        try:
            with open("ydata_profiling_report.html", "r", encoding="utf-8") as f:
                report_html = f.read()
            html(report_html, height=1000,width=800, scrolling=True)  

        except FileNotFoundError:
            st.error("Report file not found. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


    elif  st.session_state['page'] == "Data Preprocessing":
        st.markdown(DataContent.Data_preprocessing)
        pre_df_file = r"C:\Profit prediction\data\dataset\1M_data.csv"
        st.session_state.pre_df = pd.read_csv(pre_df_file)
        st.write("### Preprocessed Data Preview (First 15 Rows)")
        data_table = DataTable(df=st.session_state.pre_df.head(15))
        data_table.display_table()


    elif st.session_state['page'] == "Exploratory Data Analysis":
        file_path = r"C:\Profit prediction\data\dataset\1M_data.csv"
        st.session_state.df = pd.read_csv(file_path)
        st.header("Data Visualization")
        visualizer = DataVisualizer()
        
        plot_type = st.selectbox("Select Visualization", 
            ["Items_vs_profit_plot", "Region_plot"])
        
        if plot_type == "Items_vs_profit_plot":
            with st.spinner("Generating plot according to items..."):
                fig = visualizer.plot_profit_vs_item(st.session_state.df)
                st.plotly_chart(fig, use_container_width=False)
        
        elif plot_type == "Region_plot":
            with st.spinner("Generating Plots..."):
                fig = visualizer.plot_profit_by_region(st.session_state.df)
                st.plotly_chart(fig, use_container_width=False)
        
    
    elif st.session_state['page'] == "Machine Learning Models Used":
        st.markdown(DataContent.ml_models)
        st.image(r"C:\Profit prediction\image.png", caption="Total Profit Time Series Plot",  width=400)
        st.image(r"C:\Profit prediction\avg_acg.png", caption="Auto Correlation Plot of average  profit per month and year", width=400)
        df_metrics = pd.read_csv(r"C:\Profit prediction\data\dataset\1M_data.csv")
        data_table = DataTable(df=df_metrics)
        data_table.display_table()
        st.markdown(DataContent.best_model)

    
    elif st.session_state['page'] == "Model Predictions":
        if os.path.exists("LSTM.py"):
            # Load the LSTM module dynamically
            spec = importlib.util.spec_from_file_location("LSTM", "LSTM.py")
            lstm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lstm_module)
        
        else:
            st.error("LSTM.py file not found. Please ensure it is in the project directory.")
    else:
        st.error("Please check the code")

if __name__ == "__main__":
    main()