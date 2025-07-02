import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.lda_model
from collections import Counter

class TopicModelingModule:
    def __init__(self):
        self.vectorizer = None
        self.lda_model = None
        self.document_term_matrix = None
        self.feature_names = None
    
    def run_topic_modeling(self, processed_data):
        """Run topic modeling analysis"""
        st.header("🎯 Topic Modeling Analysis")
        
        if not processed_data:
            st.warning("No processed text data available. Please run text preprocessing first.")
            return
        
        # Column selection
        available_columns = list(processed_data.keys())
        selected_column = st.selectbox(
            "Select text column for topic modeling:",
            available_columns
        )
        
        if not selected_column:
            return
        
        # Get processed texts
        texts = processed_data[selected_column]['processed']
        
        # Filter out empty texts
        texts = [text for text in texts if text.strip()]
        
        if len(texts) < 2:
            st.error("Need at least 2 non-empty documents for topic modeling.")
            return
        
        # Topic modeling parameters
        st.subheader("⚙️ Topic Modeling Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_topics = st.slider("Number of topics:", 2, 20, 5)
            vectorizer_type = st.selectbox("Vectorizer type:", ["TF-IDF", "Count"])
        
        with col2:
            max_features = st.slider("Maximum features:", 100, 5000, 1000)
            min_df = st.slider("Minimum document frequency:", 1, 10, 2)
        
        with col3:
            max_df = st.slider("Maximum document frequency (%):", 50, 100, 95) / 100
            ngram_range = st.selectbox("N-gram range:", ["(1,1)", "(1,2)", "(1,3)"])
        
        # Parse ngram_range
        ngram_start, ngram_end = map(int, ngram_range.strip("()").split(","))
        
        if st.button("Run Topic Modeling"):
            # Vectorize texts
            if vectorizer_type == "TF-IDF":
                self.vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_range=(ngram_start, ngram_end),
                    stop_words='english'
                )
            else:
                self.vectorizer = CountVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_range=(ngram_start, ngram_end),
                    stop_words='english'
                )
            
            try:
                self.document_term_matrix = self.vectorizer.fit_transform(texts)
                self.feature_names = self.vectorizer.get_feature_names_out()
                
                # Fit LDA model
                self.lda_model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=100,
                    learning_method='batch'
                )
                
                self.lda_model.fit(self.document_term_matrix)
                
                # Display results
                self._display_topic_results(texts, n_topics, selected_column)
                
                # Topic distribution analysis
                self._analyze_topic_distribution(texts, selected_column)
                
                # Document-topic assignment
                self._show_document_topic_assignment(texts, selected_column)
                
            except Exception as e:
                st.error(f"Error in topic modeling: {str(e)}")
    
    def _display_topic_results(self, texts, n_topics, column_name):
        """Display topic modeling results"""
        st.subheader(f"📊 Topic Modeling Results - {column_name}")
        
        # Topic words
        st.write("**Topics and their top words:**")
        
        topics_data = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            topics_data.append({
                'Topic': f"Topic {topic_idx + 1}",
                'Top Words': ', '.join(top_words[:5]),
                'All Top Words': top_words,
                'Weights': top_weights
            })
            
            # Display individual topic
            with st.expander(f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Word importance bar chart
                    fig = px.bar(
                        x=top_weights,
                        y=top_words,
                        orientation='h',
                        title=f"Top Words in Topic {topic_idx + 1}",
                        labels={'x': 'Weight', 'y': 'Words'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Topic word cloud
                    word_freq = dict(zip(top_words, top_weights))
                    self._generate_topic_wordcloud(word_freq, f"Topic {topic_idx + 1}")
        
        # Topics summary table
        summary_df = pd.DataFrame(topics_data)[['Topic', 'Top Words']]
        st.dataframe(summary_df)
    
    def _generate_topic_wordcloud(self, word_freq, topic_name):
        """Generate word cloud for a specific topic"""
        if word_freq:
            wordcloud = WordCloud(
                width=400,
                height=300,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_freq)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(topic_name, fontsize=12)
            st.pyplot(fig)
            plt.close()
    
    def _analyze_topic_distribution(self, texts, column_name):
        """Analyze distribution of topics across documents"""
        st.subheader(f"📈 Topic Distribution Analysis - {column_name}")
        
        # Get document-topic probabilities
        doc_topic_probs = self.lda_model.transform(self.document_term_matrix)
        
        # Topic prevalence
        topic_prevalence = doc_topic_probs.mean(axis=0)
        topic_names = [f"Topic {i+1}" for i in range(len(topic_prevalence))]
        
        # Topic prevalence chart
        fig = px.bar(
            x=topic_names,
            y=topic_prevalence,
            title="Topic Prevalence Across All Documents",
            labels={'x': 'Topics', 'y': 'Average Probability'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Topic distribution heatmap
        st.write("**Document-Topic Probability Heatmap (Sample)**")
        sample_size = min(50, len(doc_topic_probs))
        sample_indices = np.random.choice(len(doc_topic_probs), sample_size, replace=False)
        sample_probs = doc_topic_probs[sample_indices]
        
        fig = px.imshow(
            sample_probs.T,
            title=f"Topic Distribution Across {sample_size} Sample Documents",
            labels={'x': 'Documents', 'y': 'Topics', 'color': 'Probability'},
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Topic coherence and diversity metrics
        self._calculate_topic_metrics(doc_topic_probs, topic_names)
    
    def _calculate_topic_metrics(self, doc_topic_probs, topic_names):
        """Calculate and display topic quality metrics"""
        st.write("**Topic Quality Metrics**")
        
        # Topic diversity (how spread out the topics are)
        topic_diversity = 1 - np.std(doc_topic_probs.mean(axis=0))
        
        # Document concentration (how focused documents are on specific topics)
        max_topic_probs = doc_topic_probs.max(axis=1)
        avg_concentration = max_topic_probs.mean()
        
        # Topic separation (how distinct topics are)
        topic_correlations = np.corrcoef(doc_topic_probs.T)
        avg_correlation = np.mean(topic_correlations[np.triu_indices_from(topic_correlations, k=1)])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Topic Diversity", f"{topic_diversity:.3f}", help="Higher values indicate more balanced topic distribution")
        
        with col2:
            st.metric("Avg. Document Concentration", f"{avg_concentration:.3f}", help="Higher values indicate documents are more focused on specific topics")
        
        with col3:
            st.metric("Avg. Topic Correlation", f"{avg_correlation:.3f}", help="Lower values indicate more distinct topics")
    
    def _show_document_topic_assignment(self, texts, column_name):
        """Show document-topic assignments"""
        st.subheader(f"📝 Document-Topic Assignment - {column_name}")
        
        # Get document-topic probabilities
        doc_topic_probs = self.lda_model.transform(self.document_term_matrix)
        
        # Assign primary topic to each document
        primary_topics = doc_topic_probs.argmax(axis=1)
        primary_topic_probs = doc_topic_probs.max(axis=1)
        
        # Create assignment dataframe
        assignment_df = pd.DataFrame({
            'Document_Index': range(len(texts)),
            'Primary_Topic': [f"Topic {i+1}" for i in primary_topics],
            'Confidence': primary_topic_probs,
            'Document_Preview': [text[:100] + "..." if len(text) > 100 else text for text in texts]
        })
        
        # Topic assignment distribution
        topic_counts = assignment_df['Primary_Topic'].value_counts()
        
        fig = px.pie(
            values=topic_counts.values,
            names=topic_counts.index,
            title="Distribution of Primary Topic Assignments"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show sample assignments
        st.write("**Sample Document-Topic Assignments**")
        
        # Filter and sort by confidence
        high_confidence = assignment_df[assignment_df['Confidence'] > 0.3].sort_values('Confidence', ascending=False)
        
        if not high_confidence.empty:
            display_df = high_confidence.head(20)[['Primary_Topic', 'Confidence', 'Document_Preview']]
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No high-confidence topic assignments found. Consider adjusting the number of topics.")
        
        # Confidence distribution
        fig = px.histogram(
            assignment_df,
            x='Confidence',
            title="Distribution of Topic Assignment Confidence",
            labels={'x': 'Confidence Score', 'y': 'Number of Documents'},
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def export_topic_results(self):
        """Export topic modeling results"""
        if self.lda_model is None:
            st.warning("No topic model available for export.")
            return
        
        st.subheader("📤 Export Topic Results")
        
        # Prepare export data
        export_data = {
            'topics': [],
            'document_assignments': []
        }
        
        # Export topics
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            top_weights = [float(topic[i]) for i in top_words_idx]
            
            export_data['topics'].append({
                'topic_id': topic_idx + 1,
                'words': top_words,
                'weights': top_weights
            })
        
        # Convert to downloadable format
        topics_df = pd.DataFrame([
            {
                'Topic_ID': t['topic_id'],
                'Top_Words': ', '.join(t['words'][:5]),
                'All_Words': ', '.join(t['words']),
                'Weights': ', '.join([f"{w:.4f}" for w in t['weights']])
            }
            for t in export_data['topics']
        ])
        
        csv = topics_df.to_csv(index=False)
        st.download_button(
            label="Download Topic Results as CSV",
            data=csv,
            file_name="topic_modeling_results.csv",
            mime="text/csv"
        )
