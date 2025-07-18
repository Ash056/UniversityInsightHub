import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tempfile
from fpdf import FPDF
import os
# import plotly.io as pio  <- REMOVED: No longer needed

class TopicModelingModule:
    def __init__(self):
        self.vectorizer = None
        self.lda_model = None
        self.document_term_matrix = None
        self.feature_names = None
        # For PDF export
        self.pdf_images = []
        self.pdf_descriptions = []
        self.pdf_assignments_df = None

    def run_topic_modeling(self, processed_data):
        st.header("🎯 Topic Modeling Analysis")

        # Reset PDF data for each run
        self.pdf_images = []
        self.pdf_descriptions = []
        self.pdf_assignments_df = None

        if not processed_data:
            st.warning("No processed text data available. Please run text preprocessing first.")
            return

        available_columns = list(processed_data.keys())
        selected_column = st.selectbox(
            "Select text column for topic modeling:",
            available_columns
        )

        if not selected_column:
            return

        texts = processed_data[selected_column]['processed']
        texts = [text for text in texts if text.strip()]
        if len(texts) < 2:
            st.error("Need at least 2 non-empty documents for topic modeling.")
            return

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
        ngram_start, ngram_end = map(int, ngram_range.strip("()").split(","))

        if st.button("Run Topic Modeling"):
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
                self.lda_model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=100,
                    learning_method='batch'
                )
                self.lda_model.fit(self.document_term_matrix)

                self.pdf_descriptions.append(
                    f"Topic modeling was performed on column '{selected_column}' with {n_topics} topics. "
                    "The following visualizations and metrics summarize the discovered topics and their distribution."
                )

                self._display_topic_results(texts, n_topics, selected_column)
                self._analyze_topic_distribution(texts, selected_column)
                self._show_document_topic_assignment(texts, selected_column)
                st.session_state.topic_model_results = {
                    "lda_model": self.lda_model,
                    "pdf_images": self.pdf_images,
                    "pdf_descriptions": self.pdf_descriptions,
                    "pdf_assignments_df": self.pdf_assignments_df}

            except Exception as e:
                st.error(f"Error in topic modeling: {str(e)}")

        if st.session_state.get("topic_model_results") and self.lda_model is not None:
            st.subheader("📤 Export Topic Modeling Report")
            self.lda_model = st.session_state.topic_model_results["lda_model"]
            self.pdf_images = st.session_state.topic_model_results["pdf_images"]
            self.pdf_descriptions = st.session_state.topic_model_results["pdf_descriptions"]
            self.pdf_assignments_df = st.session_state.topic_model_results["pdf_assignments_df"]
            pdf_path = self._generate_pdf_report()
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF",
                    data=f.read(),
                    file_name="topic_modeling_report.pdf",
                    mime="application/pdf"
                )
            os.remove(pdf_path)

    def _display_topic_results(self, texts, n_topics, column_name):
        st.subheader(f"📊 Topic Modeling Results - {column_name}")
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
            with st.expander(f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}"):
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        x=top_weights,
                        y=top_words,
                        orientation='h',
                        title=f"Top Words in Topic {topic_idx + 1}",
                        labels={'x': 'Weight', 'y': 'Words'}
                    )
                    fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ### MODIFICATION START: Replaced pio.to_image with Matplotlib ###
                    # Generate and save Matplotlib version for PDF
                    fig_mpl, ax_mpl = plt.subplots(figsize=(7, 5))
                    ax_mpl.barh(np.array(top_words)[::-1], np.array(top_weights)[::-1])
                    ax_mpl.set_xlabel('Weight')
                    ax_mpl.set_ylabel('Words')
                    ax_mpl.set_title(f"Top Words in Topic {topic_idx + 1}")
                    fig_mpl.tight_layout()
                    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                    fig_mpl.savefig(img_path)
                    plt.close(fig_mpl)
                    self.pdf_images.append((f"Top Words in Topic {topic_idx + 1}", img_path))
                    # ### MODIFICATION END ###

                with col2:
                    word_freq = dict(zip(top_words, top_weights))
                    wc_img_path = self._generate_topic_wordcloud(word_freq, f"Topic {topic_idx + 1}")
                    if wc_img_path:
                        self.pdf_images.append((f"Word Cloud for Topic {topic_idx + 1}", wc_img_path))
        
        summary_df = pd.DataFrame(topics_data)[['Topic', 'Top Words']]
        st.dataframe(summary_df)
        self.pdf_descriptions.append("Topics and their top words:\n" + summary_df.to_string(index=False))

    def _generate_topic_wordcloud(self, word_freq, topic_name):
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
            img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            fig.savefig(img_path)
            plt.close(fig)
            return img_path
        return None

    def _analyze_topic_distribution(self, texts, column_name):
        st.subheader(f"📈 Topic Distribution Analysis - {column_name}")
        doc_topic_probs = self.lda_model.transform(self.document_term_matrix)
        topic_prevalence = doc_topic_probs.mean(axis=0)
        topic_names = [f"Topic {i+1}" for i in range(len(topic_prevalence))]
        
        fig = px.bar(
            x=topic_names,
            y=topic_prevalence,
            title="Topic Prevalence Across All Documents",
            labels={'x': 'Topics', 'y': 'Average Probability'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # ### MODIFICATION START: Replaced pio.to_image with Matplotlib ###
        fig_mpl, ax_mpl = plt.subplots(figsize=(8, 5))
        ax_mpl.bar(topic_names, topic_prevalence)
        ax_mpl.set_xlabel('Topics')
        ax_mpl.set_ylabel('Average Probability')
        ax_mpl.set_title("Topic Prevalence Across All Documents")
        plt.setp(ax_mpl.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig_mpl.tight_layout()
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig_mpl.savefig(img_path)
        plt.close(fig_mpl)
        self.pdf_images.append(("Topic Prevalence Across All Documents", img_path))
        # ### MODIFICATION END ###
        
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
        
        # ### MODIFICATION START: Replaced pio.to_image with Matplotlib ###
        fig_mpl, ax_mpl = plt.subplots(figsize=(8, 4))
        im = ax_mpl.imshow(sample_probs.T, cmap='Blues', aspect='auto')
        ax_mpl.set_xlabel('Documents (Sampled)')
        ax_mpl.set_ylabel('Topics')
        ax_mpl.set_title(f"Topic Distribution Across {sample_size} Sample Documents")
        ax_mpl.set_yticks(np.arange(len(topic_names)))
        ax_mpl.set_yticklabels(topic_names)
        fig_mpl.colorbar(im, ax=ax_mpl, label='Probability')
        fig_mpl.tight_layout()
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig_mpl.savefig(img_path)
        plt.close(fig_mpl)
        self.pdf_images.append((f"Topic Distribution Across {sample_size} Sample Documents", img_path))
        # ### MODIFICATION END ###

        self._calculate_topic_metrics(doc_topic_probs, topic_names)

    def _calculate_topic_metrics(self, doc_topic_probs, topic_names):
        st.write("**Topic Quality Metrics**")
        topic_diversity = 1 - np.std(doc_topic_probs.mean(axis=0))
        max_topic_probs = doc_topic_probs.max(axis=1)
        avg_concentration = max_topic_probs.mean()
        topic_correlations = np.corrcoef(doc_topic_probs.T)
        avg_correlation = np.mean(topic_correlations[np.triu_indices_from(topic_correlations, k=1)])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Topic Diversity", f"{topic_diversity:.3f}", help="Higher values indicate more balanced topic distribution")
        with col2:
            st.metric("Avg. Document Concentration", f"{avg_concentration:.3f}", help="Higher values indicate documents are more focused on specific topics")
        with col3:
            st.metric("Avg. Topic Correlation", f"{avg_correlation:.3f}", help="Lower values indicate more distinct topics")
        
        self.pdf_descriptions.append(
            f"Topic Diversity: {topic_diversity:.3f}\n"
            f"Avg. Document Concentration: {avg_concentration:.3f}\n"
            f"Avg. Topic Correlation: {avg_correlation:.3f}"
        )

    def _show_document_topic_assignment(self, texts, column_name):
        st.subheader(f"📝 Document-Topic Assignment - {column_name}")
        doc_topic_probs = self.lda_model.transform(self.document_term_matrix)
        primary_topics = doc_topic_probs.argmax(axis=1)
        primary_topic_probs = doc_topic_probs.max(axis=1)
        assignment_df = pd.DataFrame({
            'Document_Index': range(len(texts)),
            'Primary_Topic': [f"Topic {i+1}" for i in primary_topics],
            'Confidence': primary_topic_probs,
            'Document_Preview': [text[:100] + "..." if len(text) > 100 else text for text in texts]
        })
        topic_counts = assignment_df['Primary_Topic'].value_counts()
        
        fig = px.pie(
            values=topic_counts.values,
            names=topic_counts.index,
            title="Distribution of Primary Topic Assignments"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ### MODIFICATION START: Replaced pio.to_image with Matplotlib ###
        fig_mpl, ax_mpl = plt.subplots(figsize=(6, 6))
        ax_mpl.pie(topic_counts.values, labels=topic_counts.index, autopct='%1.1f%%', startangle=90)
        ax_mpl.axis('equal')
        ax_mpl.set_title("Distribution of Primary Topic Assignments")
        fig_mpl.tight_layout()
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig_mpl.savefig(img_path)
        plt.close(fig_mpl)
        self.pdf_images.append(("Distribution of Primary Topic Assignments", img_path))
        # ### MODIFICATION END ###

        st.write("**Sample Document-Topic Assignments**")
        high_confidence = assignment_df[assignment_df['Confidence'] > 0.3].sort_values('Confidence', ascending=False)
        if not high_confidence.empty:
            display_df = high_confidence.head(20)[['Primary_Topic', 'Confidence', 'Document_Preview']]
            st.dataframe(display_df, use_container_width=True)
            self.pdf_assignments_df = display_df
        else:
            st.info("No high-confidence topic assignments found. Consider adjusting the number of topics.")
        
        fig = px.histogram(
            assignment_df,
            x='Confidence',
            title="Distribution of Topic Assignment Confidence",
            labels={'x': 'Confidence Score', 'y': 'Number of Documents'},
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ### MODIFICATION START: Replaced pio.to_image with Matplotlib ###
        fig_mpl, ax_mpl = plt.subplots(figsize=(7, 5))
        ax_mpl.hist(assignment_df['Confidence'], bins=20, edgecolor='black')
        ax_mpl.set_xlabel('Confidence Score')
        ax_mpl.set_ylabel('Number of Documents')
        ax_mpl.set_title("Distribution of Topic Assignment Confidence")
        fig_mpl.tight_layout()
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig_mpl.savefig(img_path)
        plt.close(fig_mpl)
        self.pdf_images.append(("Distribution of Topic Assignment Confidence", img_path))
        # ### MODIFICATION END ###

    def _generate_pdf_report(self):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(0, 10, "Topic Modeling Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(5)

        for desc in self.pdf_descriptions:
            pdf.multi_cell(0, 10, desc)
            pdf.ln(2)

        for caption, img_path in self.pdf_images:
            if pdf.get_y() > 220: # Add new page if image won't fit
                 pdf.add_page()
            pdf.set_font("Arial", 'B', size=12)
            pdf.cell(0, 10, caption, ln=True)
            pdf.image(img_path, w=170)
            pdf.ln(2)

        if self.pdf_assignments_df is not None:
            pdf.add_page()
            pdf.set_font("Arial", 'B', size=12)
            pdf.cell(0, 10, "Sample Document-Topic Assignments:", ln=True)
            pdf.set_font("Arial", size=9)
            for idx, row in self.pdf_assignments_df.iterrows():
                pdf.multi_cell(0, 5, f"{row['Primary_Topic']} | Confidence: {row['Confidence']:.2f} | {row['Document_Preview']}")
                pdf.ln(1)

        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_pdf.name)

        for _, img_path in self.pdf_images:
            try:
                os.remove(img_path)
            except Exception:
                pass
        return temp_pdf.name