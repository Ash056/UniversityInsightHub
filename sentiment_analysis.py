import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import re
from data_handler import DataHandler

class SentimentAnalysisModule:
    def __init__(self):
        self.data_handler = DataHandler()
        
    def run_sentiment_analysis(self, df):
        """Run comprehensive sentiment analysis"""
        st.header("😊 Sentiment Analysis")
        
        # Identify text columns
        text_columns = self.data_handler.identify_text_columns(df)
        
        if not text_columns:
            st.warning("No suitable text columns found for sentiment analysis.")
            return
        
        # Column selection
        st.subheader("📝 Text Column Selection")
        selected_columns = st.multiselect(
            "Select text columns for sentiment analysis:",
            text_columns,
            default=text_columns[:2] if len(text_columns) >= 2 else text_columns
        )
        
        if not selected_columns:
            st.info("Please select at least one text column for analysis.")
            return
        
        # Analysis options
        st.subheader("⚙️ Analysis Options")
        col1, col2 = st.columns(2)
        
        with col1:
            include_subjectivity = st.checkbox("Include subjectivity analysis", value=True)
            filter_neutral = st.checkbox("Filter out neutral sentiments", value=False)
        
        with col2:
            min_text_length = st.slider("Minimum text length for analysis:", 5, 100, 10)
            sentiment_threshold = st.slider("Sentiment classification threshold:", 0.0, 0.5, 0.1)
        
        if st.button("Analyze Sentiment"):
            for column in selected_columns:
                self._analyze_column_sentiment(
                    df, column, include_subjectivity, filter_neutral,
                    min_text_length, sentiment_threshold
                )
    
    def _analyze_column_sentiment(self, df, column, include_subjectivity=True, 
                                 filter_neutral=False, min_text_length=10, 
                                 sentiment_threshold=0.1):
        """Analyze sentiment for a specific column"""
        st.subheader(f"📊 Sentiment Analysis Results - {column}")
        
        # Extract and filter text data
        text_data = df[column].dropna().astype(str)
        text_data = text_data[text_data.str.len() >= min_text_length]
        
        if text_data.empty:
            st.warning(f"No text data available for analysis in column '{column}' after filtering.")
            return
        
        # Perform sentiment analysis
        results = self._calculate_sentiments(text_data)
        
        # Classify sentiments
        results['sentiment_label'] = results.apply(
            lambda row: self._classify_sentiment(row['polarity'], sentiment_threshold), axis=1
        )
        
        # Filter neutral if requested
        if filter_neutral:
            results = results[results['sentiment_label'] != 'Neutral']
        
        if results.empty:
            st.warning("No data remaining after filtering.")
            return
        
        # Display results
        self._display_sentiment_overview(results, column)
        self._display_sentiment_distribution(results, column)
        self._display_detailed_analysis(results, column, include_subjectivity)
        self._show_sentiment_examples(results, text_data, column)
        
        # Additional analyses
        self._analyze_sentiment_trends(results, column)
        if include_subjectivity:
            self._analyze_subjectivity_patterns(results, column)
    
    def _calculate_sentiments(self, text_data):
        """Calculate sentiment scores for text data"""
        results = []
        
        progress_bar = st.progress(0)
        total_texts = len(text_data)
        
        for i, text in enumerate(text_data):
            try:
                blob = TextBlob(text)
                
                results.append({
                    'text': text,
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity,
                    'text_length': len(text),
                    'word_count': len(text.split())
                })
                
            except Exception as e:
                # Handle any text processing errors
                results.append({
                    'text': text,
                    'polarity': 0.0,
                    'subjectivity': 0.0,
                    'text_length': len(text),
                    'word_count': len(text.split())
                })
            
            # Update progress
            progress_bar.progress((i + 1) / total_texts)
        
        progress_bar.empty()
        return pd.DataFrame(results)
    
    def _classify_sentiment(self, polarity, threshold):
        """Classify sentiment based on polarity score"""
        if polarity > threshold:
            return 'Positive'
        elif polarity < -threshold:
            return 'Negative'
        else:
            return 'Neutral'
    
    def _display_sentiment_overview(self, results, column):
        """Display sentiment analysis overview"""
        st.write(f"**Sentiment Overview - {column}**")
        
        # Calculate metrics
        total_responses = len(results)
        avg_polarity = results['polarity'].mean()
        avg_subjectivity = results['subjectivity'].mean()
        
        sentiment_counts = results['sentiment_label'].value_counts()
        positive_pct = (sentiment_counts.get('Positive', 0) / total_responses) * 100
        negative_pct = (sentiment_counts.get('Negative', 0) / total_responses) * 100
        neutral_pct = (sentiment_counts.get('Neutral', 0) / total_responses) * 100
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Responses", total_responses)
        with col2:
            st.metric("Avg. Polarity", f"{avg_polarity:.3f}")
        with col3:
            st.metric("Positive %", f"{positive_pct:.1f}%")
        with col4:
            st.metric("Negative %", f"{negative_pct:.1f}%")
        with col5:
            st.metric("Neutral %", f"{neutral_pct:.1f}%")
    
    def _display_sentiment_distribution(self, results, column):
        """Display sentiment distribution visualizations"""
        st.write(f"**Sentiment Distribution - {column}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment counts pie chart
            sentiment_counts = results['sentiment_label'].value_counts()
            colors = {'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#4682B4'}
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Polarity distribution histogram
            fig = px.histogram(
                results,
                x='polarity',
                title="Polarity Score Distribution",
                labels={'x': 'Polarity Score', 'y': 'Count'},
                nbins=30
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", 
                         annotation_text="Neutral")
            st.plotly_chart(fig, use_container_width=True)
        
        # Polarity vs Subjectivity scatter plot
        fig = px.scatter(
            results,
            x='subjectivity',
            y='polarity',
            color='sentiment_label',
            title="Sentiment Polarity vs Subjectivity",
            labels={'x': 'Subjectivity', 'y': 'Polarity'},
            color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#4682B4'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_detailed_analysis(self, results, column, include_subjectivity):
        """Display detailed sentiment analysis"""
        st.write(f"**Detailed Analysis - {column}**")
        
        # Statistics by sentiment
        sentiment_stats = results.groupby('sentiment_label').agg({
            'polarity': ['count', 'mean', 'std', 'min', 'max'],
            'subjectivity': ['mean', 'std'],
            'text_length': ['mean', 'std'],
            'word_count': ['mean', 'std']
        }).round(3)
        
        sentiment_stats.columns = ['_'.join(col).strip() for col in sentiment_stats.columns]
        st.dataframe(sentiment_stats)
        
        if include_subjectivity:
            # Subjectivity analysis
            st.write("**Subjectivity Analysis**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Subjectivity distribution
                fig = px.histogram(
                    results,
                    x='subjectivity',
                    color='sentiment_label',
                    title="Subjectivity Distribution by Sentiment",
                    labels={'x': 'Subjectivity Score', 'y': 'Count'},
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot of subjectivity by sentiment
                fig = px.box(
                    results,
                    x='sentiment_label',
                    y='subjectivity',
                    title="Subjectivity by Sentiment Category",
                    color='sentiment_label'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _show_sentiment_examples(self, results, original_text, column):
        """Show examples of different sentiment categories"""
        st.write(f"**Sentiment Examples - {column}**")
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            sentiment_data = results[results['sentiment_label'] == sentiment]
            
            if not sentiment_data.empty:
                with st.expander(f"{sentiment} Examples ({len(sentiment_data)} total)"):
                    # Show most extreme examples
                    if sentiment == 'Positive':
                        examples = sentiment_data.nlargest(5, 'polarity')
                    elif sentiment == 'Negative':
                        examples = sentiment_data.nsmallest(5, 'polarity')
                    else:  # Neutral
                        examples = sentiment_data.iloc[
                            sentiment_data['polarity'].abs().argsort()[:5]
                        ]
                    
                    for idx, row in examples.iterrows():
                        st.write(f"**Polarity: {row['polarity']:.3f}** | {row['text'][:200]}...")
    
    def _analyze_sentiment_trends(self, results, column):
        """Analyze sentiment trends and patterns"""
        st.write(f"**Sentiment Patterns - {column}**")
        
        # Relationship between text length and sentiment
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                results,
                x='sentiment_label',
                y='text_length',
                title="Text Length by Sentiment",
                color='sentiment_label'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                results,
                x='sentiment_label',
                y='word_count',
                title="Word Count by Sentiment",
                color='sentiment_label'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        correlations = results[['polarity', 'subjectivity', 'text_length', 'word_count']].corr()
        
        fig = px.imshow(
            correlations,
            title="Correlation Matrix: Sentiment Metrics",
            color_continuous_scale="RdBu",
            aspect="auto",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _analyze_subjectivity_patterns(self, results, column):
        """Analyze subjectivity patterns in detail"""
        st.write(f"**Subjectivity Patterns - {column}**")
        
        # Classify subjectivity levels
        results['subjectivity_level'] = pd.cut(
            results['subjectivity'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Objective', 'Moderate', 'Subjective']
        )
        
        # Subjectivity level distribution
        subj_counts = results['subjectivity_level'].value_counts()
        
        fig = px.bar(
            x=subj_counts.index,
            y=subj_counts.values,
            title="Distribution of Subjectivity Levels",
            labels={'x': 'Subjectivity Level', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cross-tabulation of sentiment and subjectivity
        crosstab = pd.crosstab(results['sentiment_label'], results['subjectivity_level'])
        
        fig = px.imshow(
            crosstab.values,
            x=crosstab.columns,
            y=crosstab.index,
            title="Sentiment vs Subjectivity Cross-tabulation",
            color_continuous_scale="Blues",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def export_sentiment_results(self, results, column):
        """Export sentiment analysis results"""
        st.subheader(f"📤 Export Sentiment Results - {column}")
        
        if results is not None and not results.empty:
            # Prepare export dataframe
            export_df = results[['text', 'polarity', 'subjectivity', 'sentiment_label']].copy()
            export_df.columns = ['Text', 'Polarity', 'Subjectivity', 'Sentiment_Label']
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label=f"Download Sentiment Results for {column}",
                data=csv,
                file_name=f"sentiment_analysis_{column}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No sentiment results available for export.")
