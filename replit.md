# University Survey Analytics Tool

## Overview

This is a comprehensive Streamlit-based web application designed for analyzing university survey data. The system provides role-based access control and includes modules for exploratory data analysis (EDA), natural language processing (NLP), topic modeling, sentiment analysis, and advanced visualizations. The application is built with a modular architecture to handle various aspects of survey data analytics.

## System Architecture

The application follows a modular, component-based architecture:

- **Frontend**: Streamlit web interface with role-based access control
- **Backend**: Python-based modular processing system
- **Authentication**: Simple hash-based authentication with role permissions
- **Data Processing**: Pandas-based data handling with validation
- **Analytics**: Specialized modules for different types of analysis

## Key Components

### 1. Authentication System (`auth.py`)
- Simple username/password authentication with predefined roles
- Role-based permission system with four user types:
  - **Dean**: Full access including user management
  - **Administrator**: Full analysis access except user management
  - **Analyst**: Advanced analysis access (no data upload/export)
  - **Viewer**: Limited access to basic EDA and sentiment analysis
- Hardcoded credentials for demo purposes (not production-ready)

### 2. Data Handler (`data_handler.py`)
- Supports CSV and Excel file uploads
- Performs data validation and quality checks
- Handles column name standardization
- Identifies duplicate rows and missing values
- Classifies columns by data type (numeric, categorical, text)

### 3. NLP Module (`nlp_module.py`)
- Text preprocessing pipeline using NLTK
- Stopword removal with custom survey-specific terms
- Lemmatization and tokenization
- Named entity recognition capabilities
- Word frequency analysis and word cloud generation

### 4. Sentiment Analysis (`sentiment_analysis.py`)
- Uses TextBlob for sentiment polarity and subjectivity analysis
- Interactive column selection for text analysis
- Configurable sentiment thresholds and filtering options
- Comprehensive sentiment distribution visualizations

### 5. Topic Modeling (`topic_modeling.py`)
- Implements Latent Dirichlet Allocation (LDA)
- TF-IDF and Count vectorization options
- K-means clustering for document grouping
- Interactive topic visualization with pyLDAvis
- Configurable number of topics and modeling parameters

### 6. Visualization Module (`visualization.py`)
- Advanced interactive visualizations using Plotly
- Multiple visualization types:
  - Interactive dashboards
  - Multi-dimensional analysis
  - Comparative analysis
  - Trend analysis
  - Custom visualization builder

### 7. Utilities (`utils.py`)
- Custom CSS styling for enhanced UI
- Common utility functions for data export/import
- Reusable UI components

## Data Flow

1. **Authentication**: User logs in with role-based credentials
2. **Data Upload**: Authorized users upload CSV/Excel files
3. **Data Validation**: System performs quality checks and preprocessing
4. **Analysis Selection**: Users choose analysis modules based on permissions
5. **Processing**: Selected modules process data according to user parameters
6. **Visualization**: Results displayed through interactive charts and dashboards
7. **Export**: Authorized users can export processed data and reports

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas/Numpy**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static plotting

### NLP Libraries
- **NLTK**: Natural language processing toolkit
- **TextBlob**: Sentiment analysis and text processing
- **WordCloud**: Word cloud generation

### Machine Learning
- **Scikit-learn**: Topic modeling and clustering algorithms
- **pyLDAvis**: LDA model visualization

### File Handling
- **openpyxl**: Excel file processing

## Deployment Strategy

The application is designed for Replit deployment with the following considerations:

- **Environment**: Python 3.8+ with pip package management
- **Entry Point**: `streamlit run app.py`
- **Dependencies**: All required packages should be listed in requirements.txt
- **Data Storage**: File-based uploads (no persistent database required)
- **Authentication**: In-memory session state management

### Deployment Steps
1. Install dependencies via pip
2. Ensure NLTK data is downloaded on first run
3. Start Streamlit server
4. Configure port forwarding for web access

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- July 02, 2025. Initial setup