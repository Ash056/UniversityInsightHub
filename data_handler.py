import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import warnings

class DataHandler:
    def __init__(self):
        self.required_columns = []
        self.data_types = {}
    
    def load_data(self, uploaded_file):
        """Load data from uploaded file with validation"""
        try:
            # Determine file type and load accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
            
            # Basic validation
            if df.empty:
                raise ValueError("The uploaded file is empty.")
            
            if len(df.columns) == 0:
                raise ValueError("No columns found in the data.")
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
            
            # Data quality check
            self._perform_quality_checks(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _perform_quality_checks(self, df):
        """Perform basic data quality checks"""
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"Found {duplicates} duplicate rows in the dataset.")
        
        # Check for missing values
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_percentage[missing_percentage > 50]
        # if not high_missing.empty:
            # st.warning(f"Columns with >50% missing values: {high_missing.index.tolist()}")
    
    def identify_text_columns(self, df):
        """Identify potential text columns for NLP analysis"""
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains substantial text (average length > 20 characters)
                avg_length = df[col].dropna().astype(str).str.len().mean()
                if avg_length > 20:
                    text_columns.append(col)
        return text_columns
    def identify_datetime_columns(self, df, threshold=0.8):
        """
        Identify columns that are likely to be datetime columns.
        Returns a list of column names.
        """
        datetime_cols = []
        for col in df.columns:
            # If already datetime dtype, add directly
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-01 00:00')
                datetime_cols.append(col)
            # If object, try to parse and check if most values are valid dates
            elif df[col].dtype == 'object':
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    # If at least `threshold` of non-null values are parsed as dates, consider as datetime
                    if parsed.notnull().mean() > threshold:
                        df[col] = df[col].dt.strftime('%Y-%m-01 00:00')
                        datetime_cols.append(col)
                except Exception:
                    pass
        return datetime_cols
    def identify_numeric_columns(self, df):
        """Identify numeric columns for statistical analysis"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def identify_categorical_columns(self, df):
        """Identify categorical columns"""
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_values = df[col].nunique()
                # Consider as categorical if less than 20% unique values or less than 50 unique values
                if unique_values < 50 or (unique_values / len(df)) < 0.2:
                    categorical_cols.append(col)
        return categorical_cols
    
    def clean_data(self, df, remove_duplicates=True, handle_missing='none'):
        """Clean the dataset based on user preferences"""
        cleaned_df = df.copy()
        
        if remove_duplicates:
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_rows = initial_rows - len(cleaned_df)
            if removed_rows > 0:
                st.info(f"Removed {removed_rows} duplicate rows.")
        
        if handle_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
        elif handle_missing == 'fill_numeric':
            numeric_cols = self.identify_numeric_columns(cleaned_df)
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif handle_missing == 'fill_categorical':
            categorical_cols = self.identify_categorical_columns(cleaned_df)
            for col in categorical_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown')
        
        return cleaned_df
    
    def filter_data(self, df, filters):
        """Apply filters to the dataset"""
        filtered_df = df.copy()
        
        for column, condition in filters.items():
            if column in df.columns:
                if condition['type'] == 'range' and df[column].dtype in ['int64', 'float64']:
                    filtered_df = filtered_df[
                        (filtered_df[column] >= condition['min']) & 
                        (filtered_df[column] <= condition['max'])
                    ]
                elif condition['type'] == 'categorical':
                    filtered_df = filtered_df[filtered_df[column].isin(condition['values'])]
                elif condition['type'] == 'text_contains':
                    filtered_df = filtered_df[
                        filtered_df[column].str.contains(condition['text'], case=False, na=False)
                    ]
        
        return filtered_df
