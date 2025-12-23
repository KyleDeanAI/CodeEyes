"""
CodeEyes: Semantic Search App for Codebases

This Streamlit application allows users to upload zipped codebases and perform
semantic search across the extracted files using language model embeddings
and FAISS indexing.

Features:
- Upload and unzip Python/text/JSON/Markdown/CSV files
- Visualize file type distributions and metadata
- Enable "Smart Sampling" to build an efficient, targeted vector index by:
    - Extracting frequent terms from a small random sample
    - Indexing only chunks related to those terms
- Perform similarity-based search using BAAI/bge-small-en-v1.5 embeddings
- Display matching code snippets and enable full file download

Key Technologies:
- Streamlit for UI
- LangChain for embedding and document handling
- FAISS for fast vector similarity search
- Plotly for interactive charts

Kyle Dean Bauer
"""

import os
import shutil
import stat
import time
import zipfile
import random
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def safe_remove_directory(directory_path):
    """Remove a directory, handling read-only files and Windows file locks."""
    if not os.path.exists(directory_path):
        return

    def remove_readonly(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    try:
        shutil.rmtree(directory_path, onexc=remove_readonly)
    except OSError:
        time.sleep(0.1)
        try:
            shutil.rmtree(directory_path, onexc=remove_readonly)
        except OSError:
            try:
                trash_path = f"{directory_path}_trash_{int(time.time())}"
                os.rename(directory_path, trash_path)
            except OSError:
                pass

class CodeEyesApp:
    """Main application class handling UI, file processing, and search logic."""

    def __init__(self):
        """Initialize the application state and configuration."""
        st.set_page_config(page_title="CodeEyes", layout="wide")
        self._init_session_state()
        self._apply_styles()

    @staticmethod
    def _init_session_state():
        """Initialize Streamlit session state variables."""
        if "query" not in st.session_state:
            st.session_state["query"] = ""
        if "probe_terms" not in st.session_state:
            st.session_state["probe_terms"] = []

    @staticmethod
    def _apply_styles():
        """Inject custom CSS styles."""
        st.markdown("""
            <style>
                .main { background-color: #0e1117; color: #ffffff; }
                .block-container { padding-top: 2rem; padding-bottom: 2rem; }
                h1, h2, h3, h4, h5, h6 { color: #00c0ff; }
                .stButton>button { background-color: #00c0ff; color: black; font-weight: bold;
                 transition: 0.3s ease-in-out; }
                .stButton>button:hover { transform: scale(1.05); }
                .stTextInput>div>div>input { background-color: #1e1e1e; color: white; }
                .codeeyes-logo {
                    font-family: cursive; font-size: 2.5rem; color: #d3d3d3;
                    background-color: transparent; padding: 0.2em 0.6em; border-radius: 10px;
                    position: absolute; top: 20px; left: 50%; transform: translateX(-50%);
                    z-index: 10; animation: fadeInDown 4s ease-in-out;
                     text-shadow: 2px 2px 4px #000000;
                }
                @keyframes fadeInDown {
                    0% { opacity: 0; transform: translate(-50%, -30px); }
                    100% { opacity: 1; transform: translate(-50%, 0); }
                }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def load_files(root_dir):
        """Walk through directories and load compatible text files into a DataFrame."""
        data_list = []
        for root, _, files in os.walk(root_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".py", ".txt", ".md", ".json", ".csv"):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f_reader:
                            content_str = f_reader.read()
                        data_list.append({
                            "path": fpath,
                            "content": content_str,
                            "lines": len(content_str.splitlines())
                        })
                    except (UnicodeDecodeError, PermissionError):
                        continue
        return pd.DataFrame(data_list)

    @staticmethod
    def extract_probe_terms(contents, top_k=20):
        """Helper to extract common terms for the sidebar visualization."""
        tokens = []
        for content in contents:
            tokens += re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", content)
        common = Counter(tokens).most_common(top_k)
        return [term for term, _ in common]

    def smart_sample(self, df, embedder):
        """Generate a vector store using a random sample of terms to filter chunks."""
        if "probe_terms" in st.session_state and st.session_state["probe_terms"]:
            p_terms = st.session_state["probe_terms"]
        else:
            sample = df['content'].sample(frac=0.1, random_state=42).tolist()
            p_terms = self.extract_probe_terms(sample)
            st.session_state["probe_terms"] = p_terms

        split = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        final_docs = []

        pb_sample = st.progress(0, text="Smart Indexing...")
        for i, (_, row) in enumerate(df.iterrows()):
            chunks = split.split_text(row['content'])
            for chk in chunks:
                if any(term in chk for term in p_terms):
                    final_docs.append(Document(page_content=chk, metadata={"source": row['path']}))
            pb_sample.progress((i + 1) / len(df))

        pb_sample.empty()
        return FAISS.from_documents(final_docs, embedder)

    def build_store(self, df, use_smart=True):
        """Orchestrate the creation of the FAISS vector store."""
        emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        if use_smart:
            return self.smart_sample(df, emb)

        split = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        all_docs = []
        pb_full = st.progress(0, text="Full Indexing...")

        for i, (_, row) in enumerate(df.iterrows()):
            all_docs.extend([
                Document(page_content=c, metadata={"source": row['path']})
                for c in split.split_text(row['content'])
            ])
            pb_full.progress((i + 1) / len(df))

        pb_full.empty()
        return FAISS.from_documents(all_docs, emb)

    def render_sidebar(self):
        """Render sidebar elements and handle file uploads."""
        st.sidebar.header("Code Upload")
        uploaded_file = st.sidebar.file_uploader("Upload ZIP:", type="zip")

        if uploaded_file is None:
            st.session_state.pop("vectorstore", None)
            st.session_state["query"] = ""
            return pd.DataFrame()

        if uploaded_file.size > 200 * 1024 * 1024:
            st.sidebar.error("File too large (200MB limit).")
            st.stop()

        with open("uploaded_code.zip", "wb") as f_writer:
            f_writer.write(uploaded_file.getbuffer())

        safe_remove_directory("uploaded_files")

        try:
            with zipfile.ZipFile("uploaded_code.zip", 'r') as z_ref:
                z_ref.extractall("uploaded_files")
            st.sidebar.success("Files extracted!")
        except (zipfile.BadZipFile, OSError) as e:
            st.sidebar.error(f"ZIP error: {e}")
            st.stop()

        return self.load_files("uploaded_files")

    def render_visualizations(self, uploaded_df, smart_val):
        """Render charts and metrics based on uploaded data."""
        if uploaded_df.empty:
            return

        col1, col2 = st.columns(2)
        col1.metric("Files", len(uploaded_df))
        col2.metric("Total Lines", uploaded_df['lines'].sum())

        f_types = uploaded_df['path'].apply(lambda x: x.split('.')[-1]).value_counts().reset_index()
        f_types.columns = ['type', 'count']
        type_fig = px.bar(f_types, x='type', y='count', title='File Types', template='plotly_dark')
        st.plotly_chart(type_fig, width="stretch")

        if smart_val:
            # Re-calculate or use cached probe terms
            if not st.session_state.get("probe_terms"):
                sample_texts = uploaded_df['content'].sample(frac=0.1, random_state=42).tolist()
                st.session_state["probe_terms"] = self.extract_probe_terms(sample_texts)

            with st.sidebar.expander("Probe Terms Sampled", expanded=True):
                st.caption(
                    "These are the most common terms extracted from a random 10% sample"
                    " of your codebase. "
                    "They're used to guide which chunks get indexed when Smart Sampling is enabled."
                )

                all_tokens = [
                    t for t in uploaded_df['content'].sample(frac=0.1, random_state=42).tolist()
                    for t in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", t)
                ]
                term_frequencies = Counter(all_tokens)
                probe_freqs = [term_frequencies[term] for term in st.session_state["probe_terms"]]
                freq_df = pd.DataFrame({"term": st.session_state["probe_terms"],
                                        "count": probe_freqs})

                st.markdown("#### Probe Terms Sampled")
                term_list_html = "<div style='padding: 10px; line-height: 1.8; font-size: 16px;'>"
                term_list_html += ", ".join(f"<code>{term}</code>"
                                            for term in st.session_state["probe_terms"])
                term_list_html += "</div>"
                st.markdown(term_list_html, unsafe_allow_html=True)

                fig = px.bar(freq_df, x="term", y="count", title="Probe Term Frequencies",
                             template="plotly_dark")
                fig.update_layout(height=500, margin={"t": 40, "b": 40}, xaxis_title="",
                                  yaxis_title="")
                fig.update_traces(marker_line_width=1.5)
                st.plotly_chart(fig, width="stretch")

    def run(self):
        """Main execution entry point."""
        st.markdown("<div class='codeeyes-logo'>CodeEyes</div>", unsafe_allow_html=True)
        st.image("https://imgur.com/CvKsjK4.png", width="stretch")

        taglines = [
            "Code navigation through semantic search",
            "Let your code speak back to you",
            "Find insights faster with AI-powered search",
            "From ZIP to insight in seconds",
            "Code. Index. Query. Done."
        ]
        st.markdown("<h1 style='text-align: center;'>Search Your Codebase</h1>",
                    unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:gray;'>{random.choice(taglines)}</p>",
                    unsafe_allow_html=True)

        uploaded_df = self.render_sidebar()
        smart_val = st.sidebar.checkbox("Smart Sampling", value=True)

        if not uploaded_df.empty:
            self.render_visualizations(uploaded_df, smart_val)

            if st.sidebar.button("Build Index"):
                st.session_state["vectorstore"] = self.build_store(uploaded_df, use_smart=smart_val)
                st.sidebar.success("Index Ready!")

        # Search Interface
        if "vectorstore" in st.session_state:
            q_in = st.text_input("Search Code:", key="q_input")
            if st.button("Search") or st.session_state["query"]:
                st.session_state["query"] = q_in or st.session_state["query"]
                hits = st.session_state["vectorstore"].similarity_search(st.session_state["query"],
                                                                         k=5)

                for idx, hit in enumerate(hits):
                    with st.expander(f"{idx + 1}. {hit.metadata['source']}"):
                        st.code(hit.page_content[:500])
                        try:
                            with open(hit.metadata['source'], "rb") as f_data:
                                st.download_button(
                                    "Download", f_data.read(),
                                    os.path.basename(hit.metadata['source']),
                                    key=f"d_{idx}"
                                )
                        except OSError:
                            st.warning("Locked or missing.")

        st.markdown(
            "<div style='text-align: center; color: gray; margin-top: 5em;'>CodeEyes 2025</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    app = CodeEyesApp()
    app.run()
