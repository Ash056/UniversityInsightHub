import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from data_handler import DataHandler

class VisualizationModule:
    def __init__(self):
        self.data_handler = DataHandler()
    
    def create_advanced_visualizations(self, df):
        """Create advanced interactive visualizations"""
        st.header("📈 Advanced Visualizations")
        
        # Visualization type selection
        viz_types = [
            "Interactive Dashboard",
            "Multi-dimensional Analysis", 
            "Comparative Analysis",
            "Trend Analysis",
            "Custom Visualization Builder"
        ]
        
        selected_viz = st.selectbox("Select visualization type:", viz_types)
        
        if selected_viz == "Interactive Dashboard":
            self._create_interactive_dashboard(df)
        elif selected_viz == "Multi-dimensional Analysis":
            self._create_multidimensional_analysis(df)
        elif selected_viz == "Comparative Analysis":
            self._create_comparative_analysis(df)
        elif selected_viz == "Trend Analysis":
            self._create_trend_analysis(df)
        elif selected_viz == "Custom Visualization Builder":
            self._create_custom_viz_builder(df)
    
    def _create_interactive_dashboard(self, df):
        """Create an interactive dashboard with key metrics"""
        st.subheader("📊 Interactive Dashboard")
        
        # Get column types
        numeric_cols = self.data_handler.identify_numeric_columns(df)
        categorical_cols = self.data_handler.identify_categorical_columns(df)
        
        if not numeric_cols and not categorical_cols:
            st.warning("No suitable columns found for dashboard creation.")
            return
        
        # Dashboard configuration
        col1, col2 = st.columns(2)
        
        with col1:
            if numeric_cols:
                primary_metric = st.selectbox("Primary numeric metric:", numeric_cols)
            else:
                primary_metric = None
        
        with col2:
            if categorical_cols:
                grouping_var = st.selectbox("Grouping variable:", ["None"] + categorical_cols)
                if grouping_var == "None":
                    grouping_var = None
            else:
                grouping_var = None
        
        # Create dashboard
        if primary_metric:
            # Summary statistics
            st.write("**Key Metrics**")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric("Mean", f"{df[primary_metric].mean():.2f}")
            with metric_cols[1]:
                st.metric("Median", f"{df[primary_metric].median():.2f}")
            with metric_cols[2]:
                st.metric("Std Dev", f"{df[primary_metric].std():.2f}")
            with metric_cols[3]:
                st.metric("Count", f"{df[primary_metric].count()}")
            
            # Main visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution plot
                if grouping_var:
                    fig = px.histogram(
                        df, x=primary_metric, color=grouping_var,
                        title=f"Distribution of {primary_metric} by {grouping_var}",
                        marginal="box"
                    )
                else:
                    fig = px.histogram(
                        df, x=primary_metric,
                        title=f"Distribution of {primary_metric}",
                        marginal="box"
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot or bar chart
                if grouping_var:
                    if df[grouping_var].nunique() <= 20:
                        fig = px.box(
                            df, x=grouping_var, y=primary_metric,
                            title=f"{primary_metric} by {grouping_var}"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                    else:
                        # For many categories, show top categories
                        top_categories = df[grouping_var].value_counts().head(15).index
                        filtered_df = df[df[grouping_var].isin(top_categories)]
                        fig = px.box(
                            filtered_df, x=grouping_var, y=primary_metric,
                            title=f"{primary_metric} by Top 15 {grouping_var}"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                else:
                    # Quantile plot
                    sorted_values = df[primary_metric].dropna().sort_values()
                    quantiles = np.linspace(0, 1, len(sorted_values))
                    fig = px.line(
                        x=quantiles, y=sorted_values,
                        title=f"Quantile Plot: {primary_metric}",
                        labels={'x': 'Quantile', 'y': primary_metric}
                    )
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualizations for categorical data
        if categorical_cols:
            st.write("**Categorical Analysis**")
            
            selected_cat = st.selectbox("Select categorical variable for analysis:", categorical_cols)
            
            if selected_cat:
                value_counts = df[selected_cat].value_counts()
                
                # Limit to top categories for clarity
                if len(value_counts) > 15:
                    value_counts = value_counts.head(15)
                    st.info(f"Showing top 15 categories out of {df[selected_cat].nunique()} total.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig = px.bar(
                        x=value_counts.index, y=value_counts.values,
                        title=f"Distribution of {selected_cat}",
                        labels={'x': selected_cat, 'y': 'Count'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig = px.pie(
                        values=value_counts.values, names=value_counts.index,
                        title=f"Proportion of {selected_cat}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _create_multidimensional_analysis(self, df):
        """Create multidimensional analysis visualizations"""
        st.subheader("🎯 Multi-dimensional Analysis")
        
        numeric_cols = self.data_handler.identify_numeric_columns(df)
        categorical_cols = self.data_handler.identify_categorical_columns(df)
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for multi-dimensional analysis.")
            return
        
        # Variable selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("X-axis variable:", numeric_cols)
        with col2:
            y_var = st.selectbox("Y-axis variable:", [col for col in numeric_cols if col != x_var])
        with col3:
            color_var = st.selectbox("Color variable:", ["None"] + categorical_cols + numeric_cols)
            if color_var == "None":
                color_var = None
        
        # Additional dimensions
        size_var = st.selectbox("Size variable (optional):", ["None"] + numeric_cols)
        if size_var == "None":
            size_var = None
        
        # Create scatter plot
        fig = px.scatter(
            df, x=x_var, y=y_var, color=color_var, size=size_var,
            title=f"Multi-dimensional Analysis: {x_var} vs {y_var}",
            hover_data=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )
        
        # Add trend line
        if st.checkbox("Add trend line"):
            fig.add_trace(
                go.Scatter(
                    x=df[x_var], y=np.poly1d(np.polyfit(df[x_var].dropna(), df[y_var].dropna(), 1))(df[x_var]),
                    mode='lines', name='Trend Line', line=dict(color='red', dash='dash')
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix for selected variables
        if len(numeric_cols) >= 3:
            st.write("**Correlation Analysis**")
            selected_vars = st.multiselect(
                "Select variables for correlation analysis:",
                numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
            )
            
            if len(selected_vars) >= 2:
                correlation_matrix = df[selected_vars].corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 3D visualization if available
        if len(numeric_cols) >= 3:
            st.write("**3D Visualization**")
            z_var = st.selectbox("Z-axis variable:", [col for col in numeric_cols if col not in [x_var, y_var]])
            
            fig = px.scatter_3d(
                df, x=x_var, y=y_var, z=z_var, color=color_var,
                title=f"3D Scatter Plot: {x_var}, {y_var}, {z_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_comparative_analysis(self, df):
        """Create comparative analysis visualizations"""
        st.subheader("⚖️ Comparative Analysis")
        
        numeric_cols = self.data_handler.identify_numeric_columns(df)
        categorical_cols = self.data_handler.identify_categorical_columns(df)
        
        if not categorical_cols:
            st.warning("Need categorical variables for comparative analysis.")
            return
        
        # Comparison setup
        compare_by = st.selectbox("Compare by:", categorical_cols)
        
        # Limit categories for meaningful comparison
        categories = df[compare_by].value_counts()
        if len(categories) > 10:
            st.info(f"Showing top 10 categories out of {len(categories)} total.")
            top_categories = categories.head(10).index
            comparison_df = df[df[compare_by].isin(top_categories)].copy()
        else:
            comparison_df = df.copy()
        
        # Numeric variable comparison
        if numeric_cols:
            st.write("**Numeric Variable Comparison**")
            
            selected_numeric = st.multiselect(
                "Select numeric variables to compare:",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_numeric:
                # Box plots
                for var in selected_numeric:
                    fig = px.box(
                        comparison_df, x=compare_by, y=var,
                        title=f"Distribution of {var} by {compare_by}"
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics table
                summary_stats = comparison_df.groupby(compare_by)[selected_numeric].agg(['mean', 'median', 'std', 'count']).round(2)
                st.write("**Summary Statistics by Group**")
                st.dataframe(summary_stats)
        
        # Categorical variable comparison
        other_categorical = [col for col in categorical_cols if col != compare_by]
        if other_categorical:
            st.write("**Categorical Variable Comparison**")
            
            selected_cat = st.selectbox("Select categorical variable to compare:", other_categorical)
            
            # Create contingency table
            contingency = pd.crosstab(comparison_df[compare_by], comparison_df[selected_cat])
            
            # Stacked bar chart
            fig = px.bar(
                contingency, x=contingency.index, y=contingency.columns,
                title=f"Distribution of {selected_cat} by {compare_by}",
                labels={'x': compare_by, 'y': 'Count'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Normalized stacked bar chart
            contingency_norm = contingency.div(contingency.sum(axis=1), axis=0)
            fig = px.bar(
                contingency_norm, x=contingency_norm.index, y=contingency_norm.columns,
                title=f"Normalized Distribution of {selected_cat} by {compare_by}",
                labels={'x': compare_by, 'y': 'Proportion'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_trend_analysis(self, df):
        """Create trend analysis visualizations"""
        st.subheader("📈 Trend Analysis")
        
        # Look for date/time columns
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].head())
                    date_cols.append(col)
                except:
                    pass
        
        numeric_cols = self.data_handler.identify_numeric_columns(df)
        
        if not numeric_cols:
            st.warning("No numeric columns found for trend analysis.")
            return
        
        if date_cols:
            st.write("**Time-based Trend Analysis**")
            
            # Date column selection
            date_col = st.selectbox("Select date column:", date_cols)
            metric_col = st.selectbox("Select metric for trend analysis:", numeric_cols)
            
            # Convert to datetime
            try:
                df_trend = df.copy()
                df_trend[date_col] = pd.to_datetime(df_trend[date_col])
                df_trend = df_trend.dropna(subset=[date_col, metric_col])
                df_trend = df_trend.sort_values(date_col)
                
                # Time-based aggregation
                agg_level = st.selectbox("Aggregation level:", ["Daily", "Weekly", "Monthly"])
                
                if agg_level == "Daily":
                    df_trend['period'] = df_trend[date_col].dt.date
                elif agg_level == "Weekly":
                    df_trend['period'] = df_trend[date_col].dt.to_period('W')
                elif agg_level == "Monthly":
                    df_trend['period'] = df_trend[date_col].dt.to_period('M')
                
                # Aggregate data
                trend_data = df_trend.groupby('period')[metric_col].agg(['mean', 'count', 'std']).reset_index()
                
                # Line plot
                fig = px.line(
                    trend_data, x='period', y='mean',
                    title=f"{agg_level} Trend of {metric_col}",
                    labels={'mean': f'Average {metric_col}', 'period': 'Time Period'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add confidence intervals if enough data
                if not trend_data['std'].isna().all():
                    fig = go.Figure()
                    
                    # Main line
                    fig.add_trace(go.Scatter(
                        x=trend_data['period'], y=trend_data['mean'],
                        mode='lines+markers', name=f'Average {metric_col}'
                    ))
                    
                    # Confidence interval
                    upper_bound = trend_data['mean'] + trend_data['std']
                    lower_bound = trend_data['mean'] - trend_data['std']
                    
                    fig.add_trace(go.Scatter(
                        x=trend_data['period'], y=upper_bound,
                        mode='lines', line=dict(width=0), showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=trend_data['period'], y=lower_bound,
                        mode='lines', line=dict(width=0),
                        fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                        name='±1 Std Dev'
                    ))
                    
                    fig.update_layout(title=f"{agg_level} Trend with Confidence Interval")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing date column: {str(e)}")
        
        else:
            st.write("**Index-based Trend Analysis**")
            st.info("No date columns detected. Creating index-based trends.")
            
            # Select variables for trend analysis
            selected_vars = st.multiselect(
                "Select variables for trend analysis:",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_vars:
                # Create index-based trends
                df_indexed = df[selected_vars].reset_index()
                
                # Rolling averages
                window_size = st.slider("Rolling average window:", 5, 50, 10)
                
                fig = go.Figure()
                
                for var in selected_vars:
                    # Original data
                    fig.add_trace(go.Scatter(
                        x=df_indexed.index, y=df_indexed[var],
                        mode='lines', name=var, opacity=0.5
                    ))
                    
                    # Rolling average
                    rolling_avg = df_indexed[var].rolling(window=window_size).mean()
                    fig.add_trace(go.Scatter(
                        x=df_indexed.index, y=rolling_avg,
                        mode='lines', name=f'{var} (MA-{window_size})',
                        line=dict(width=3)
                    ))
                
                fig.update_layout(
                    title="Trend Analysis with Moving Averages",
                    xaxis_title="Index",
                    yaxis_title="Values"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _create_custom_viz_builder(self, df):
        """Create custom visualization builder"""
        st.subheader("🛠️ Custom Visualization Builder")
        
        # Chart type selection
        chart_types = [
            "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
            "Box Plot", "Violin Plot", "Heatmap", "Sunburst Chart"
        ]
        
        chart_type = st.selectbox("Select chart type:", chart_types)
        
        # Get available columns
        numeric_cols = self.data_handler.identify_numeric_columns(df)
        categorical_cols = self.data_handler.identify_categorical_columns(df)
        all_cols = list(df.columns)
        
        # Dynamic parameter selection based on chart type
        if chart_type == "Scatter Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols + categorical_cols)
            with col2:
                y_col = st.selectbox("Y-axis:", numeric_cols)
            with col3:
                color_col = st.selectbox("Color by:", ["None"] + all_cols)
            
            if color_col == "None":
                color_col = None
            
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                           title=f"Custom Scatter Plot: {x_col} vs {y_col}")
        
        elif chart_type == "Line Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis:", all_cols)
            with col2:
                y_col = st.selectbox("Y-axis:", numeric_cols)
            
            fig = px.line(df, x=x_col, y=y_col, 
                         title=f"Custom Line Chart: {x_col} vs {y_col}")
        
        elif chart_type == "Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Category:", categorical_cols + all_cols)
            with col2:
                y_col = st.selectbox("Value:", ["Count"] + numeric_cols)
            
            if y_col == "Count":
                value_counts = df[x_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Count of {x_col}")
            else:
                fig = px.bar(df, x=x_col, y=y_col,
                           title=f"Custom Bar Chart: {x_col} vs {y_col}")
        
        elif chart_type == "Histogram":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Variable:", numeric_cols)
            with col2:
                bins = st.slider("Number of bins:", 10, 100, 30)
            
            fig = px.histogram(df, x=x_col, nbins=bins,
                             title=f"Histogram of {x_col}")
        
        elif chart_type == "Box Plot":
            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Numeric variable:", numeric_cols)
            with col2:
                x_col = st.selectbox("Group by:", ["None"] + categorical_cols)
            
            if x_col == "None":
                fig = px.box(df, y=y_col, title=f"Box Plot of {y_col}")
            else:
                fig = px.box(df, x=x_col, y=y_col, 
                           title=f"Box Plot: {y_col} by {x_col}")
        
        elif chart_type == "Heatmap":
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect("Select numeric columns:", numeric_cols,
                                             default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)
                if selected_cols:
                    correlation_matrix = df[selected_cols].corr()
                    fig = px.imshow(correlation_matrix, title="Custom Correlation Heatmap",
                                  color_continuous_scale="RdBu", aspect="auto", text_auto=True)
                else:
                    fig = None
            else:
                st.warning("Need at least 2 numeric columns for heatmap.")
                fig = None
        
        else:  # Default to simple chart
            fig = px.bar(df.head(), title="Sample Chart")
        
        # Display the chart
        if fig:
            # Customization options
            with st.expander("Chart Customization"):
                col1, col2 = st.columns(2)
                with col1:
                    title = st.text_input("Chart Title:", value=fig.layout.title.text or "Custom Chart")
                    width = st.slider("Chart Width:", 400, 1200, 800)
                with col2:
                    height = st.slider("Chart Height:", 300, 800, 500)
                    theme = st.selectbox("Color Theme:", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])
            
            # Apply customizations
            fig.update_layout(
                title=title,
                width=width,
                height=height,
                template=theme
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export option
            if st.button("Export Chart as HTML"):
                html_str = fig.to_html()
                st.download_button(
                    label="Download Chart HTML",
                    data=html_str,
                    file_name="custom_chart.html",
                    mime="text/html"
                )
