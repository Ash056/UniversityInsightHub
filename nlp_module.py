import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import matplotlib.pyplot as plt
from data_handler import DataHandler

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

class NLPModule:
    def __init__(self):
        self.data_handler = DataHandler()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add custom stop words for survey data
        self.stop_words.update([
            'survey', 'question', 'answer', 'response', 'respondent',
            'university', 'college', 'student', 'faculty', 'course'
        ])
    
    def run_preprocessing(self, df):
        """Run NLP preprocessing pipeline"""
        st.header("🔤 Text Processing & NLP Analysis")
        
        # Identify text columns
        text_columns = self.data_handler.identify_text_columns(df)
        
        if not text_columns:
            st.warning("No suitable text columns found for NLP analysis.")
            return None
        
        # Column selection
        st.subheader("📝 Text Column Selection")
        selected_columns = st.multiselect(
            "Select text columns for NLP analysis:",
            text_columns,
            default=text_columns[:3] if len(text_columns) >= 3 else text_columns
        )
        
        if not selected_columns:
            st.info("Please select at least one text column for analysis.")
            return None
        
        # Preprocessing options
        st.subheader("⚙️ Preprocessing Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            remove_stopwords = st.checkbox("Remove stopwords", value=True)
            apply_lemmatization = st.checkbox("Apply lemmatization", value=True)
        
        with col2:
            min_word_length = st.slider("Minimum word length", 1, 10, 3)
            remove_numbers = st.checkbox("Remove numbers", value=True)
        
        with col3:
            custom_stopwords = st.text_area(
                "Additional stopwords (comma-separated):",
                placeholder="word1, word2, word3"
            )
        # Use session_state to persist processed data
        if "processed_data" not in st.session_state:
            st.session_state.processed_data = None
            st.session_state.processed_columns = None
            
        
        if st.button("Process Text"):
            # Add custom stopwords
            if custom_stopwords:
                additional_stops = [word.strip().lower() for word in custom_stopwords.split(',')]
                self.stop_words.update(additional_stops)
            
            # Process each selected column
            processed_data = {}
            
            for col in selected_columns:
                st.write(f"**Processing column: {col}**")
                
                # # Extract and clean text
                text_data = df[col].dropna().astype(str)
                
                # # Basic statistics
                # self._show_text_statistics(text_data, col)
                
                # Preprocess text
                processed_text = self._preprocess_text_column(
                    text_data,
                    remove_stopwords=remove_stopwords,
                    apply_lemmatization=apply_lemmatization,
                    min_word_length=min_word_length,
                    remove_numbers=remove_numbers
                )
                
                processed_data[col] = {
                    'original': text_data,
                    'processed': processed_text,
                    'tokens': self._tokenize_texts(processed_text)
                }
            # Store in session_state
            st.session_state.processed_data = processed_data
            st.session_state.processed_columns = selected_columns
        # If processed data exists, show analysis widgets
        if st.session_state.processed_data:
            for col in st.session_state.processed_columns:
                st.write(f"**Processing column: {col}**")
                # Basic statistics

                text_data = df[col].dropna().astype(str)
                self._show_text_statistics(text_data, col)
                # Word frequency analysis
                self._show_word_frequency_analysis(st.session_state.processed_data[col]['tokens'], col)
                
                # Word cloud
                self._generate_word_cloud(st.session_state.processed_data[col]['tokens'], col)
                
                # N-gram analysis
                self._show_ngram_analysis(st.session_state.processed_data[col]['tokens'], col)
            
            return st.session_state.processed_data
        
        return None
    
    def _show_text_statistics(self, text_data, column_name):
        """Show basic text statistics"""
        st.write(f"**Text Statistics for {column_name}**")
        
        # Calculate statistics
        total_responses = len(text_data)
        avg_length = text_data.str.len().mean()
        median_length = text_data.str.len().median()
        total_words = text_data.str.split().str.len().sum()
        avg_words = text_data.str.split().str.len().mean()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Responses", total_responses)
        with col2:
            st.metric("Avg. Characters", f"{avg_length:.1f}")
        with col3:
            st.metric("Median Characters", f"{median_length:.1f}")
        with col4:
            st.metric("Total Words", f"{total_words:,}")
        with col5:
            st.metric("Avg. Words", f"{avg_words:.1f}")
        
        # Length distribution
        lengths = text_data.str.len()
        fig = px.histogram(
            x=lengths,
            title=f"Text Length Distribution - {column_name}",
            labels={'x': 'Character Count', 'y': 'Frequency'},
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _preprocess_text_column(self, text_series, remove_stopwords=True, 
                               apply_lemmatization=True, min_word_length=3, 
                               remove_numbers=True):
        """Preprocess a column of text data"""
        processed_texts = []
        
        progress_bar = st.progress(0)
        total_texts = len(text_series)
        
        for i, text in enumerate(text_series):
            processed_text = self._preprocess_single_text(
                text, remove_stopwords, apply_lemmatization, 
                min_word_length, remove_numbers
            )
            processed_texts.append(processed_text)
            
            # Update progress
            progress_bar.progress((i + 1) / total_texts)
        
        progress_bar.empty()
        return processed_texts
    
    def _preprocess_single_text(self, text, remove_stopwords=True, 
                               apply_lemmatization=True, min_word_length=3, 
                               remove_numbers=True):
        """Preprocess a single text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove numbers if requested
        if remove_numbers:
            tokens = [token for token in tokens if not token.isdigit()]
        
        # Remove short words
        tokens = [token for token in tokens if len(token) >= min_word_length]
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply lemmatization
        if apply_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def _tokenize_texts(self, processed_texts):
        """Tokenize processed texts into individual words"""
        all_tokens = []
        for text in processed_texts:
            tokens = word_tokenize(text)
            all_tokens.extend(tokens)
        return all_tokens
    
    def _show_word_frequency_analysis(self, tokens, column_name):
        """Show word frequency analysis"""
        st.write(f"**Word Frequency Analysis - {column_name}**")
        
        # Calculate word frequencies
        word_freq = Counter(tokens)
        
        # Top words
        top_n = st.slider(f"Number of top words to show for {column_name}:", 5, 50, 20, key=f"top_words_{column_name}")
        top_words = word_freq.most_common(top_n)
        
        if top_words:
            words, frequencies = zip(*top_words)
            
            # Bar chart
            fig = px.bar(
                x=list(frequencies),
                y=list(words),
                orientation='h',
                title=f"Top {top_n} Most Frequent Words - {column_name}",
                labels={'x': 'Frequency', 'y': 'Words'}
            )
            fig.update_layout(height=max(400, top_n * 20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Frequency table
            freq_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            freq_df['Percentage'] = (freq_df['Frequency'] / len(tokens) * 100).round(2)
            st.dataframe(freq_df)
    
    def _generate_word_cloud(self, tokens, column_name):
        """Generate and display word cloud"""
        st.write(f"**Word Cloud - {column_name}**")
        
        if tokens:
            # Create word frequency dictionary
            word_freq = Counter(tokens)
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate_from_frequencies(word_freq)
            
            # Display using matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Word Cloud - {column_name}', fontsize=16, pad=20)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No words available for word cloud generation.")
    
    def _show_ngram_analysis(self, tokens, column_name):
        """Show n-gram analysis"""
        st.write(f"**N-gram Analysis - {column_name}**")
        
        if len(tokens) < 2:
            st.info("Not enough tokens for n-gram analysis.")
            return
        
        # N-gram selection
        ngram_type = st.selectbox(
            f"Select n-gram type for {column_name}:",
            ['Bigrams (2-words)', 'Trigrams (3-words)'],
            key=f"ngram_{column_name}"
        )
        
        n = 2 if ngram_type == 'Bigrams (2-words)' else 3
        
        # Generate n-grams
        ngrams = self._generate_ngrams(tokens, n)
        ngram_freq = Counter(ngrams)
        
        if ngram_freq:
            top_ngrams = ngram_freq.most_common(15)
            
            if top_ngrams:
                ngram_labels = [' '.join(ngram) for ngram, _ in top_ngrams]
                frequencies = [freq for _, freq in top_ngrams]
                
                # Bar chart
                fig = px.bar(
                    x=frequencies,
                    y=ngram_labels,
                    orientation='h',
                    title=f"Top 15 {ngram_type} - {column_name}",
                    labels={'x': 'Frequency', 'y': f'{ngram_type}'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No {ngram_type.lower()} found.")
    
    def _generate_ngrams(self, tokens, n):
        """Generate n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        return ngrams
    
    def extract_named_entities(self, text_data, column_name):
        """Extract named entities from text data"""
        st.write(f"**Named Entity Recognition - {column_name}**")
        
        entities = {}
        sample_size = min(100, len(text_data))  # Limit for performance
        
        progress_bar = st.progress(0)
        
        for i, text in enumerate(text_data.head(sample_size)):
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            
            # Extract entities
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    entity_name = ' '.join([token for token, pos in chunk.leaves()])
                    entity_type = chunk.label()
                    
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(entity_name)
            
            progress_bar.progress((i + 1) / sample_size)
        
        progress_bar.empty()
        
        # Display results
        if entities:
            for entity_type, entity_list in entities.items():
                entity_freq = Counter(entity_list)
                st.write(f"**{entity_type} Entities:**")
                top_entities = entity_freq.most_common(10)
                
                if top_entities:
                    entity_df = pd.DataFrame(top_entities, columns=['Entity', 'Frequency'])
                    st.dataframe(entity_df)
        else:
            st.info("No named entities found in the sample.")
