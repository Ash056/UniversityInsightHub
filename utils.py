import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import base64
from datetime import datetime
import os

class Utils:
    @staticmethod
    def load_css():
        """Load custom CSS styles"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
        
        .warning-card {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .success-card {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .info-card {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_dataframe_info(df, title="Dataset Information"):
        """Display comprehensive dataframe information"""
        st.subheader(title)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
        
        # Data types summary
        with st.expander("Column Details"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2),
                'Unique': df.nunique(),
                'Unique %': (df.nunique() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
    
    @staticmethod
    def create_download_link(data, filename, file_type="csv"):
        """Create download link for data"""
        if file_type == "csv":
            if isinstance(data, pd.DataFrame):
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
            else:
                b64 = base64.b64encode(str(data).encode()).decode()
                href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">Download file</a>'
        elif file_type == "json":
            json_str = json.dumps(data, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download JSON file</a>'
        
        return href
    
    @staticmethod
    def safe_division(numerator, denominator, default=0):
        """Safely divide two numbers"""
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except:
            return default
    
    @staticmethod
    def format_large_number(num):
        """Format large numbers with appropriate suffixes"""
        if abs(num) >= 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        elif abs(num) >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif abs(num) >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return str(num)
    
    @staticmethod
    def validate_data_quality(df):
        """Validate data quality and return issues"""
        issues = []
        
        # Check for empty dataframe
        if df.empty:
            issues.append("Dataset is empty")
            return issues
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)")
        
        # Check for columns with all missing values
        all_missing = df.columns[df.isnull().all()].tolist()
        if all_missing:
            issues.append(f"Columns with all missing values: {all_missing}")
        
        # Check for columns with high missing values (>50%)
        high_missing = df.columns[df.isnull().sum() / len(df) > 0.5].tolist()
        if high_missing:
            issues.append(f"Columns with >50% missing values: {high_missing}")
        
        # Check for constant columns
        constant_cols = df.columns[df.nunique() <= 1].tolist()
        if constant_cols:
            issues.append(f"Constant columns (no variation): {constant_cols}")
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
            if len(outliers) > len(df) * 0.1:  # More than 10% outliers
                outlier_cols.append(col)
        
        if outlier_cols:
            issues.append(f"Columns with many outliers (>10%): {outlier_cols}")
        
        return issues
    
    @staticmethod
    def clean_column_names(df):
        """Clean column names for better processing"""
        df = df.copy()
        
        # Convert to lowercase and replace spaces with underscores
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^\w]', '', regex=True)
        
        # Remove duplicate column names
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        
        return df
    
    @staticmethod
    def get_memory_usage(df):
        """Get detailed memory usage of dataframe"""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        memory_info = {
            'total_mb': total_memory / 1024**2,
            'by_column': (memory_usage / 1024**2).to_dict(),
            'by_dtype': df.memory_usage(deep=True).groupby(df.dtypes).sum() / 1024**2
        }
        
        return memory_info
    
    @staticmethod
    def suggest_data_types(df):
        """Suggest optimal data types for columns"""
        suggestions = {}
        
        for col in df.columns:
            current_dtype = df[col].dtype
            
            # Skip if already optimal
            if current_dtype in ['category', 'bool']:
                continue
                
            # Check if can be converted to category
            if current_dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    suggestions[col] = 'category'
            
            # Check if numeric can be downcasted
            elif current_dtype in ['int64', 'float64']:
                if current_dtype == 'int64':
                    min_val, max_val = df[col].min(), df[col].max()
                    if min_val >= -128 and max_val <= 127:
                        suggestions[col] = 'int8'
                    elif min_val >= -32768 and max_val <= 32767:
                        suggestions[col] = 'int16'
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        suggestions[col] = 'int32'
                
                elif current_dtype == 'float64':
                    # Check if can be int
                    if df[col].dropna().apply(lambda x: x.is_integer()).all():
                        suggestions[col] = 'int32'
                    else:
                        suggestions[col] = 'float32'
        
        return suggestions
    
    @staticmethod
    def apply_data_type_suggestions(df, suggestions):
        """Apply suggested data type changes"""
        df_optimized = df.copy()
        
        for col, new_dtype in suggestions.items():
            try:
                if new_dtype == 'category':
                    df_optimized[col] = df_optimized[col].astype('category')
                else:
                    df_optimized[col] = df_optimized[col].astype(new_dtype)
            except Exception as e:
                st.warning(f"Could not convert {col} to {new_dtype}: {str(e)}")
        
        return df_optimized
    
    @staticmethod
    def get_sample_data(df, n_samples=1000):
        """Get a representative sample of the data"""
        if len(df) <= n_samples:
            return df
        
        # For small datasets, return all
        if len(df) < n_samples * 2:
            return df
        
        # Stratified sampling if possible
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            # Use the first categorical column for stratification
            strat_col = categorical_cols[0]
            try:
                return df.groupby(strat_col, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, n_samples // df[strat_col].nunique())))
                )
            except:
                pass
        
        # Random sampling if stratified sampling fails
        return df.sample(n=n_samples, random_state=42)
    
    @staticmethod
    def export_session_state():
        """Export current session state"""
        exportable_keys = ['data', 'processed_data', 'user_role', 'username']
        export_data = {}
        
        for key in exportable_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], pd.DataFrame):
                    export_data[key] = st.session_state[key].to_dict()
                else:
                    export_data[key] = st.session_state[key]
        
        return export_data
    
    @staticmethod
    def log_user_action(action, details=None):
        """Log user actions for audit trail"""
        timestamp = datetime.now().isoformat()
        user = st.session_state.get('username', 'anonymous')
        role = st.session_state.get('user_role', 'unknown')
        
        log_entry = {
            'timestamp': timestamp,
            'user': user,
            'role': role,
            'action': action,
            'details': details or {}
        }
        
        # In a real application, this would write to a persistent log
        if 'action_log' not in st.session_state:
            st.session_state.action_log = []
        
        st.session_state.action_log.append(log_entry)
    
    @staticmethod
    def display_help_section():
        """Display help information"""
        with st.expander("📚 Help & Documentation"):
            st.markdown("""
            ## University Survey Analytics Tool - User Guide
            
            ### Getting Started
            1. **Login**: Use your credentials to access the system
            2. **Upload Data**: Upload CSV or Excel files containing survey responses
            3. **Explore**: Navigate through different analysis tabs
            
            ### Features by Role
            
            **Dean**: Full access to all features including data upload, analysis, and export
            **Administrator**: Access to upload, analysis, and export (no user management)
            **Analyst**: Access to all analysis features (no data upload/export)
            **Viewer**: Limited access to basic EDA and sentiment analysis
            
            ### Analysis Modules
            
            **📊 Exploratory Data Analysis (EDA)**
            - Dataset overview and quality assessment
            - Statistical summaries and distributions
            - Correlation analysis
            - Interactive visualizations
            
            **🔤 Text Processing**
            - NLP preprocessing pipeline
            - Word frequency analysis
            - N-gram analysis
            - Named entity recognition
            
            **🎯 Topic Modeling**
            - Latent Dirichlet Allocation (LDA)
            - Topic distribution analysis
            - Document-topic assignments
            
            **😊 Sentiment Analysis**
            - Polarity and subjectivity scores
            - Sentiment classification
            - Detailed sentiment patterns
            
            **📈 Advanced Visualizations**
            - Multi-dimensional analysis
            - Comparative analysis
            - Custom visualization builder
            
            ### Data Requirements
            - **File formats**: CSV, Excel (.xlsx, .xls)
            - **Text columns**: For NLP analysis, ensure text responses are in separate columns
            - **Data quality**: Clean data will produce better results
            
            ### Tips for Best Results
            - Ensure text columns contain meaningful content (>20 characters on average)
            - Remove or handle missing values appropriately
            - Use descriptive column names
            - For topic modeling, aim for at least 50-100 documents
            
            ### Support
            If you encounter issues or need assistance, please contact your system administrator.
            """)
    
    @staticmethod
    def check_system_requirements():
        """Check if system meets requirements"""
        requirements = {
            'pandas_version': pd.__version__,
            'streamlit_version': st.__version__,
            'memory_available': 'Available',  # Simplified check
            'python_version': '3.7+'  # Assumed
        }
        
        return requirements
