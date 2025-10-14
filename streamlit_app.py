#!/usr/bin/env python3
"""
AutoCoEv v2 - Streamlit Web Interface

Fast protein-protein interaction prediction with ESM-2 and STRING validation.
No LLM required for cost-effective analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import yaml
from datetime import datetime
from io import StringIO, BytesIO
import sys

# Add modern directory to path
sys.path.append(str(Path(__file__).parent / "modern"))

from modern.core.fast_embedding_generator import FastProteinEmbedder
from modern.integrations.string_db import StringDBClient
from modern.report.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="AutoCoEv v2 - PPI Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'proteins' not in st.session_state:
    st.session_state.proteins = None


def parse_fasta(fasta_content: str) -> dict:
    """Parse FASTA format and return protein dictionary."""
    proteins = {}
    current_id = None
    current_seq = []

    for line in fasta_content.split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_id:
                proteins[current_id] = ''.join(current_seq)
            current_id = line[1:].split()[0]
            current_seq = []
        elif line:
            current_seq.append(line)

    if current_id:
        proteins[current_id] = ''.join(current_seq)

    return proteins


def run_analysis(proteins: dict, threshold: float, species: int,
                string_threshold: int, use_string: bool):
    """Run the complete AutoCoEv v2 analysis pipeline."""

    # Load configuration
    config_path = Path(__file__).parent / "modern" / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize components
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Initialize ESM-2
    status_text.text("Loading ESM-2 model...")
    progress_bar.progress(10)
    device = config['methods']['protein_lm'].get('device', 'auto')
    cache_dir = config.get('data', {}).get('esm_cache', './cache/esm_embeddings')
    embedding_gen = FastProteinEmbedder(cache_dir=cache_dir, device=device)

    # Step 2: Generate protein pairs
    status_text.text("Generating protein pairs...")
    progress_bar.progress(20)
    protein_ids = list(proteins.keys())
    n_proteins = len(protein_ids)
    pairs = []

    for i in range(n_proteins):
        for j in range(i + 1, n_proteins):
            pairs.append((protein_ids[i], protein_ids[j]))

    st.info(f"Analyzing {n_proteins} proteins ({len(pairs)} pairs)")

    # Step 3: Calculate coevolution scores
    status_text.text("Calculating coevolution scores with ESM-2...")
    predictions = []

    for idx, (protein_a, protein_b) in enumerate(pairs):
        progress = 20 + int((idx / len(pairs)) * 40)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing pair {idx + 1}/{len(pairs)}: {protein_a} - {protein_b}")

        seq_a = proteins[protein_a]
        seq_b = proteins[protein_b]

        score = embedding_gen.get_coevolution_score(protein_a, protein_b, seq_a, seq_b)

        if score >= threshold:
            predictions.append({
                'protein_a': protein_a,
                'protein_b': protein_b,
                'autocoev_score': score,
                'sequence_a': seq_a,
                'sequence_b': seq_b
            })

    st.success(f"Found {len(predictions)} interactions above threshold {threshold}")

    # Step 4: STRING validation
    if use_string and predictions:
        status_text.text("Validating with STRING database...")
        progress_bar.progress(70)

        string_config = config['methods']['string_db'].copy()
        string_config['species'] = species
        string_config['required_score'] = string_threshold
        string_client = StringDBClient(string_config)

        predictions = string_client.enrich_with_string(predictions)
        progress_bar.progress(90)
    else:
        # Add default STRING fields
        for pred in predictions:
            pred.update({
                'string_score': 0.0,
                'combined_confidence': pred['autocoev_score'] * 0.9,
                'is_novel': True,
                'exists_in_string': False
            })

    # Step 5: Generate report
    status_text.text("Generating report...")
    progress_bar.progress(95)

    metadata = {
        'n_proteins': n_proteins,
        'n_pairs': len(pairs),
        'threshold': threshold,
        'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    progress_bar.progress(100)
    status_text.text("Analysis complete!")

    return predictions, metadata


def create_visualizations(df: pd.DataFrame):
    """Create interactive visualizations of results."""

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distribution")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['autocoev_score'],
            name='AutoCoEv Score',
            nbinsx=20,
            marker_color='lightblue'
        ))

        if 'string_score' in df.columns and df['string_score'].sum() > 0:
            fig.add_trace(go.Histogram(
                x=df['string_score'],
                name='STRING Score',
                nbinsx=20,
                marker_color='lightcoral'
            ))

        fig.update_layout(
            barmode='overlay',
            xaxis_title='Score',
            yaxis_title='Count',
            height=400
        )
        fig.update_traces(opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confidence vs. Novelty")

        fig = px.scatter(
            df,
            x='combined_confidence',
            y='autocoev_score',
            color='is_novel',
            hover_data=['protein_a', 'protein_b'],
            labels={
                'combined_confidence': 'Combined Confidence',
                'autocoev_score': 'AutoCoEv Score',
                'is_novel': 'Novel Discovery'
            },
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Network visualization
    st.subheader("Protein Interaction Network")

    # Create network data
    nodes = set()
    edges = []

    for _, row in df.head(50).iterrows():  # Limit to top 50 for readability
        nodes.add(row['protein_a'])
        nodes.add(row['protein_b'])
        edges.append({
            'source': row['protein_a'],
            'target': row['protein_b'],
            'score': row['combined_confidence']
        })

    st.info(f"Showing top 50 interactions (total: {len(df)} interactions)")

    # Simple network display as table
    st.dataframe(
        pd.DataFrame(edges),
        use_container_width=True,
        height=300
    )


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<div class="main-header">AutoCoEv v2 - Protein Interaction Predictor</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fast PPI prediction with ESM-2 and STRING validation</div>',
                unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    threshold = st.sidebar.slider(
        "AutoCoEv Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum coevolution score to report (higher = stricter)"
    )

    use_string = st.sidebar.checkbox(
        "Enable STRING Validation",
        value=True,
        help="Validate predictions against STRING database"
    )

    if use_string:
        species = st.sidebar.selectbox(
            "Species",
            options=[
                (9606, "Human"),
                (10090, "Mouse"),
                (10116, "Rat"),
                (7227, "Fruit Fly"),
                (6239, "C. elegans"),
                (559292, "S. cerevisiae")
            ],
            format_func=lambda x: x[1],
            index=0
        )[0]

        string_threshold = st.sidebar.slider(
            "STRING Confidence Threshold",
            min_value=150,
            max_value=900,
            value=400,
            step=50,
            help="Minimum STRING confidence score (400=medium, 700=high)"
        )
    else:
        species = 9606
        string_threshold = 400

    # Performance info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Performance Info")
    st.sidebar.info("""
    **Speed**: ~1-2 seconds per protein

    **GPU**: Automatically detected

    **Memory**: ~4GB for ESM-2 model
    """)

    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.markdown("""
    AutoCoEv v2 uses:
    - **ESM-2**: Protein language model (650M params)
    - **STRING**: Protein interaction database

    **No LLM** = No API costs!

    [GitHub Repository](https://github.com/Luq16/autocoev_v2)
    """)

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Analysis", "Results", "Documentation"])

    with tab1:
        st.header("Upload Protein Sequences")

        # File upload
        uploaded_file = st.file_uploader(
            "Choose a FASTA file",
            type=['fasta', 'fa', 'faa'],
            help="Upload a FASTA file with protein sequences"
        )

        # Example data option
        col1, col2 = st.columns([3, 1])
        with col2:
            use_example = st.button("Load Example Data", type="secondary")

        if use_example:
            example_path = Path(__file__).parent / "test_data" / "test_proteins.fasta"
            if example_path.exists():
                with open(example_path) as f:
                    fasta_content = f.read()
                st.session_state.proteins = parse_fasta(fasta_content)
                st.success(f"Loaded {len(st.session_state.proteins)} example proteins")
            else:
                st.error("Example data not found. Please upload your own FASTA file.")

        if uploaded_file is not None:
            fasta_content = uploaded_file.read().decode('utf-8')
            st.session_state.proteins = parse_fasta(fasta_content)

            st.success(f"Loaded {len(st.session_state.proteins)} proteins")

            # Display proteins
            with st.expander("View Protein Sequences"):
                for protein_id, sequence in st.session_state.proteins.items():
                    st.text(f">{protein_id}\n{sequence[:100]}{'...' if len(sequence) > 100 else ''}")

        # Run analysis button
        if st.session_state.proteins:
            st.markdown("---")

            n_proteins = len(st.session_state.proteins)
            n_pairs = n_proteins * (n_proteins - 1) // 2

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Proteins", n_proteins)
            with col2:
                st.metric("Total Pairs", n_pairs)
            with col3:
                est_time = n_proteins * 2  # ~2 seconds per protein
                st.metric("Est. Time", f"{est_time}s")

            if st.button("Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Running analysis..."):
                    try:
                        predictions, metadata = run_analysis(
                            st.session_state.proteins,
                            threshold,
                            species,
                            string_threshold,
                            use_string
                        )

                        st.session_state.results = {
                            'predictions': predictions,
                            'metadata': metadata
                        }
                        st.session_state.analysis_complete = True

                        st.balloons()
                        st.success("Analysis complete! Check the Results tab.")

                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.exception(e)

    with tab2:
        st.header("Analysis Results")

        if not st.session_state.analysis_complete:
            st.info("Run an analysis to see results here.")
        else:
            results = st.session_state.results
            predictions = results['predictions']
            metadata = results['metadata']

            # Summary metrics
            st.subheader("Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Predictions", len(predictions))

            with col2:
                novel_count = sum(1 for p in predictions if p.get('is_novel', True))
                st.metric("Novel Discoveries", novel_count)

            with col3:
                known_count = len(predictions) - novel_count
                st.metric("STRING-Validated", known_count)

            with col4:
                if predictions:
                    avg_conf = sum(p['combined_confidence'] for p in predictions) / len(predictions)
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")

            # Convert to DataFrame
            df = pd.DataFrame(predictions)
            df = df.sort_values('combined_confidence', ascending=False)

            # Visualizations
            st.markdown("---")
            create_visualizations(df)

            # Detailed results table
            st.markdown("---")
            st.subheader("Detailed Results")

            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                show_novel = st.checkbox("Show Novel Only", value=False)
            with col2:
                min_confidence = st.slider(
                    "Min Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )

            # Apply filters
            filtered_df = df.copy()
            if show_novel:
                filtered_df = filtered_df[filtered_df['is_novel'] == True]
            filtered_df = filtered_df[filtered_df['combined_confidence'] >= min_confidence]

            # Display table
            display_columns = [
                'protein_a', 'protein_b', 'autocoev_score',
                'string_score', 'combined_confidence', 'is_novel'
            ]

            st.dataframe(
                filtered_df[display_columns].style.format({
                    'autocoev_score': '{:.3f}',
                    'string_score': '{:.3f}',
                    'combined_confidence': '{:.3f}'
                }).background_gradient(subset=['combined_confidence'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )

            # Download section
            st.markdown("---")
            st.subheader("Export Results")

            col1, col2 = st.columns(2)

            with col1:
                # CSV download
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"autocoev_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                # Generate markdown report
                report_gen = ReportGenerator()
                report = report_gen.generate_full_report(predictions, metadata)

                st.download_button(
                    label="Download Report (Markdown)",
                    data=report,
                    file_name=f"autocoev_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    with tab3:
        st.header("Documentation")

        st.markdown("""
        ## Quick Start Guide

        ### 1. Prepare Your Data

        Upload a FASTA file with protein sequences:

        ```
        >EGFR
        MRPSGTAGAALLALLAALCPASRALEEKKVC...
        >GRB2
        MASMTGGQQMGRDLYDDDDKDPMM...
        ```

        ### 2. Configure Parameters

        - **AutoCoEv Threshold**: Higher values = stricter predictions
          - 0.3: Loose (many predictions)
          - 0.5: Balanced (recommended)
          - 0.7: Strict (high confidence only)

        - **STRING Validation**: Validates against known interactions
          - Identifies novel vs. known interactions
          - Provides experimental evidence scores

        ### 3. Run Analysis

        Click "Run Analysis" and wait for completion (~2 seconds per protein).

        ### 4. Review Results

        - View interactive visualizations
        - Filter by confidence or novelty
        - Export results as CSV or Markdown

        ## Understanding Results

        ### Score Interpretation

        | Score | Confidence | Interpretation |
        |-------|-----------|----------------|
        | 0.9-1.0 | Very High | Strong evidence, priority follow-up |
        | 0.8-0.9 | High | Good evidence, recommended validation |
        | 0.6-0.8 | Medium | Moderate evidence, further analysis needed |
        | 0.4-0.6 | Low-Medium | Weak evidence, cautious interpretation |

        ### Novel vs. Known

        - **Novel**: Not found in STRING database - potential new discoveries
        - **Known**: Validated by STRING - provides confidence in method

        ## Performance

        ### Speed Benchmarks

        | Proteins | Pairs | Est. Time |
        |----------|-------|-----------|
        | 5 | 10 | 10 seconds |
        | 10 | 45 | 20 seconds |
        | 50 | 1,225 | 2 minutes |
        | 100 | 4,950 | 5 minutes |

        ### Hardware Requirements

        - **Minimum**: 8GB RAM, CPU
        - **Recommended**: 16GB RAM, GPU (10x speedup)
        - **Storage**: 5GB for ESM-2 model

        ## Troubleshooting

        ### GPU Not Detected

        ESM-2 will automatically use GPU if available. Falls back to CPU if not.

        ### Memory Errors

        - Increase threshold to reduce predictions
        - Process smaller protein sets
        - Use CPU mode (slower but less memory)

        ### STRING API Errors

        Rate limiting is automatic. If errors persist, disable STRING validation.

        ## Citation

        If you use AutoCoEv v2, please cite:

        - **ESM-2**: Lin et al. (2023) Science
        - **STRING**: Szklarczyk et al. (2023) NAR

        ## Support

        - [GitHub Issues](https://github.com/Luq16/autocoev_v2/issues)
        - [Documentation](https://github.com/Luq16/autocoev_v2)
        """)


if __name__ == "__main__":
    main()
