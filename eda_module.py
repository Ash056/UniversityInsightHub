import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from data_handler import DataHandler

class EDAModule:
    def __init__(self):
        self.data_handler = DataHandler()
    
    def run_analysis(self, df):
        """Run comprehensive exploratory data analysis"""
        st.header("📊 Exploratory Data Analysis")
        
        # Dataset Overview
        self._show_dataset_overview(df)
        
        # Data Quality Assessment
        self._show_data_quality(df)
        
        # Statistical Summary
        self._show_statistical_summary(df)
        
        # Distribution Analysis
        self._show_distribution_analysis(df)
        
        # Correlation Analysis
        self._show_correlation_analysis(df)
        
        # Categorical Analysis
        self._show_categorical_analysis(df)
    
    def _show_dataset_overview(self, df):
        """Show basic dataset information"""
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info)
    
    def _show_data_quality(self, df):
        """Show data quality metrics"""
        st.subheader("🔍 Data Quality Assessment")
        
        # Missing value heatmap
        if df.isnull().sum().sum() > 0:
            st.write("**Missing Values Heatmap**")
            fig = px.imshow(
                df.isnull().astype(int),
                title="Missing Values Pattern",
                color_continuous_scale="Reds",
                aspect="auto"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data completeness by column
        completeness = ((df.count() / len(df)) * 100).sort_values()
        fig = px.bar(
            x=completeness.values,
            y=completeness.index,
            orientation='h',
            title="Data Completeness by Column (%)",
            labels={'x': 'Completeness %', 'y': 'Columns'}
        )
        fig.update_layout(height=max(400, len(df.columns) * 20))
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_statistical_summary(self, df):
        """Show statistical summary for numeric columns"""
        st.subheader("📈 Statistical Summary")
        
        numeric_cols = self.data_handler.identify_numeric_columns(df)
        
        if numeric_cols:
            # Statistical summary table
            summary_stats = df[numeric_cols].describe()
            st.write("**Descriptive Statistics**")
            st.dataframe(summary_stats)
            
            # Box plot for numeric columns
            if len(numeric_cols) <= 10:  # Limit to avoid overcrowding
                fig = make_subplots(
                    rows=1, cols=len(numeric_cols),
                    subplot_titles=numeric_cols,
                    shared_yaxis=False
                )
                
                for i, col in enumerate(numeric_cols):
                    fig.add_trace(
                        go.Box(y=df[col].dropna(), name=col, showlegend=False),
                        row=1, col=i+1
                    )
                
                fig.update_layout(
                    title="Distribution of Numeric Variables (Box Plots)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for statistical analysis.")
    
    def _show_distribution_analysis(self, df):
        """Show distribution analysis for selected columns"""
        st.subheader("📊 Distribution Analysis")
        
        numeric_cols = self.data_handler.identify_numeric_columns(df)
        categorical_cols = self.data_handler.identify_categorical_columns(df)
        
        # Numeric distributions
        if numeric_cols:
            st.write("**Numeric Variable Distributions**")
            selected_numeric = st.selectbox(
                "Select numeric column for detailed distribution analysis:",
                numeric_cols,
                key="numeric_dist"
            )
            
            if selected_numeric:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(
                        df, x=selected_numeric,
                        title=f"Distribution of {selected_numeric}",
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Q-Q plot for normality check
                    from scipy import stats
                    fig = go.Figure()
                    
                    # Remove NaN values
                    data_clean = df[selected_numeric].dropna()
                    
                    # Calculate theoretical quantiles
                    sorted_data = np.sort(data_clean)
                    n = len(sorted_data)
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
                    
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sorted_data,
                        mode='markers',
                        name='Q-Q Plot'
                    ))
                    
                    # Add reference line
                    min_val = min(theoretical_quantiles.min(), sorted_data.min())
                    max_val = max(theoretical_quantiles.max(), sorted_data.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Reference Line',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Q-Q Plot: {selected_numeric}",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Categorical distributions
        if categorical_cols:
            st.write("**Categorical Variable Distributions**")
            selected_categorical = st.selectbox(
                "Select categorical column for analysis:",
                categorical_cols,
                key="categorical_dist"
            )
            
            if selected_categorical:
                value_counts = df[selected_categorical].value_counts()
                
                # Limit to top 20 categories to avoid overcrowding
                if len(value_counts) > 20:
                    value_counts = value_counts.head(20)
                    st.info(f"Showing top 20 categories out of {df[selected_categorical].nunique()} total.")
                
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution of {selected_categorical}",
                    labels={'x': selected_categorical, 'y': 'Count'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    def _show_correlation_analysis(self, df):
        """Show correlation analysis for numeric variables"""
        st.subheader("🔗 Correlation Analysis")
        
        numeric_cols = self.data_handler.identify_numeric_columns(df)
        
        if len(numeric_cols) >= 2:
            # Correlation matrix
            correlation_matrix = df[numeric_cols].corr()
            
            # Heatmap
            fig = px.imshow(
                correlation_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto",
                text_auto=True
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pairwise correlation scatter plots
            if len(numeric_cols) <= 6:  # Limit for performance
                st.write("**Pairwise Scatter Plots**")
                selected_cols = st.multiselect(
                    "Select columns for pairwise analysis:",
                    numeric_cols,
                    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                )
                
                if len(selected_cols) >= 2:
                    fig = px.scatter_matrix(
                        df[selected_cols],
                        title="Pairwise Scatter Plot Matrix",
                        height=800
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")
    
    def _show_categorical_analysis(self, df):
        """Show analysis for categorical variables"""
        st.subheader("🏷️ Categorical Analysis")
        
        categorical_cols = self.data_handler.identify_categorical_columns(df)
        
        if categorical_cols:
            # Cross-tabulation analysis
            if len(categorical_cols) >= 2:
                st.write("**Cross-tabulation Analysis**")
                col1_select = st.selectbox("Select first categorical variable:", categorical_cols, key="cat1")
                col2_select = st.selectbox("Select second categorical variable:", categorical_cols, key="cat2")
                
                if col1_select != col2_select:
                    # Create contingency table
                    contingency = pd.crosstab(df[col1_select], df[col2_select])
                    
                    # Limit categories for visualization
                    if contingency.shape[0] > 15 or contingency.shape[1] > 15:
                        st.warning("Too many categories for clear visualization. Showing top categories only.")
                        contingency = contingency.iloc[:15, :15]
                    
                    # Heatmap of contingency table
                    fig = px.imshow(
                        contingency.values,
                        x=contingency.columns,
                        y=contingency.index,
                        title=f"Cross-tabulation: {col1_select} vs {col2_select}",
                        color_continuous_scale="Blues",
                        text_auto=True
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show the actual contingency table
                    st.write("**Contingency Table**")
                    st.dataframe(contingency)
            
            # Categorical summary
            st.write("**Categorical Variables Summary**")
            cat_summary = []
            for col in categorical_cols:
                cat_summary.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                    'Most Frequent Count': df[col].value_counts().iloc[0] if not df[col].empty else 0,
                    'Percentage': (df[col].value_counts().iloc[0] / len(df) * 100).round(2) if not df[col].empty else 0
                })
            
            cat_summary_df = pd.DataFrame(cat_summary)
            st.dataframe(cat_summary_df)
        else:
            st.info("No categorical columns found for analysis.")
