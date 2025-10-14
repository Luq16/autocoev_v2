#!/usr/bin/env python3
"""
AutoCoEv v2 - Modern Implementation
Fast protein-protein interaction prediction with STRING and LLM validation

Usage:
    python autocoev_modern.py --input proteins.fasta --output results/ [options]
"""

import argparse
import logging
import yaml
import time
from pathlib import Path
from typing import List, Tuple, Dict
import sys

# Add modern modules to path
sys.path.insert(0, str(Path(__file__).parent / "modern"))

from core.fast_embedding_generator import FastProteinEmbedder
from integrations.string_db import STRINGDatabase
from integrations.llm_literature_search import LLMLiteratureSearch
from report.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoCoEvModern:
    """
    AutoCoEv v2 Modern Pipeline

    Features:
    - Fast analysis with ESM-2 (50-100x speedup)
    - Automatic STRING database validation
    - LLM-powered literature search
    - Comprehensive reporting
    """

    def __init__(self, config_path: str = None):
        """
        Initialize AutoCoEv v2 pipeline

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'methods': {
                    'protein_lm': {'enabled': True, 'device': 'auto'},
                    'string_db': {'enabled': True, 'species': 9606, 'required_score': 400},
                    'llm_validation': {'enabled': True, 'provider': 'openai', 'model': 'gpt-4'}
                },
                'scoring': {'detailed_threshold': 0.6},
                'output': {'format': 'markdown'}
            }

        logger.info("AutoCoEv v2 pipeline initialized")

    def load_proteins(self, fasta_file: str) -> List[Tuple[str, str]]:
        """
        Load proteins from FASTA file

        Args:
            fasta_file: Path to FASTA file

        Returns:
            List of (protein_id, sequence) tuples
        """
        proteins = []

        with open(fasta_file, 'r') as f:
            current_id = None
            current_seq = []

            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous protein
                    if current_id:
                        proteins.append((current_id, ''.join(current_seq)))

                    # Start new protein
                    current_id = line[1:].split()[0]  # Take first word after '>'
                    current_seq = []
                else:
                    current_seq.append(line)

            # Save last protein
            if current_id:
                proteins.append((current_id, ''.join(current_seq)))

        logger.info(f"Loaded {len(proteins)} proteins from {fasta_file}")
        return proteins

    def generate_pairs(self, proteins: List[Tuple[str, str]]) -> List[Tuple]:
        """
        Generate all pairs of proteins for analysis

        Args:
            proteins: List of (protein_id, sequence) tuples

        Returns:
            List of protein pairs
        """
        pairs = []
        for i in range(len(proteins)):
            for j in range(i + 1, len(proteins)):
                pairs.append((proteins[i], proteins[j]))

        logger.info(f"Generated {len(pairs)} protein pairs for analysis")
        return pairs

    def analyze_pairs(self, pairs: List[Tuple], threshold: float = 0.5) -> List[Dict]:
        """
        Analyze protein pairs using ESM-2

        Args:
            pairs: List of protein pairs
            threshold: Minimum score threshold

        Returns:
            List of predictions above threshold
        """
        logger.info("Analyzing protein pairs with ESM-2...")

        # Initialize ESM-2 embedder
        embedder = FastProteinEmbedder(
            device=self.config['methods']['protein_lm'].get('device', 'auto')
        )

        predictions = []
        for i, (protein_a, protein_b) in enumerate(pairs):
            pid_a, seq_a = protein_a
            pid_b, seq_b = protein_b

            # Get coevolution score
            score = embedder.get_coevolution_score(pid_a, pid_b, seq_a, seq_b)

            if score >= threshold:
                predictions.append({
                    'protein_a': pid_a,
                    'protein_b': pid_b,
                    'score': score
                })

            if (i + 1) % 100 == 0:
                logger.info(f"Analyzed {i + 1}/{len(pairs)} pairs")

        logger.info(f"Found {len(predictions)} interactions above threshold {threshold}")
        return predictions

    def validate_with_string(self, predictions: List[Dict]) -> List[Dict]:
        """
        Validate predictions with STRING database

        Args:
            predictions: List of predicted interactions

        Returns:
            Enriched predictions with STRING data
        """
        if not self.config['methods']['string_db'].get('enabled', True):
            logger.info("STRING validation disabled")
            return predictions

        logger.info("Validating predictions with STRING database...")

        string_db = STRINGDatabase(
            species=self.config['methods']['string_db'].get('species', 9606),
            required_score=self.config['methods']['string_db'].get('required_score', 400)
        )

        enriched = string_db.enrich_with_string(predictions)

        logger.info(f"STRING validation complete")
        return enriched

    def literature_search(self, predictions: List[Dict], top_n: int = 20) -> List[Dict]:
        """
        Perform LLM literature search on top predictions

        Args:
            predictions: List of predictions
            top_n: Number of top predictions to validate with LLM

        Returns:
            Predictions with LLM validation
        """
        if not self.config['methods']['llm_validation'].get('enabled', True):
            logger.info("LLM validation disabled")
            return predictions

        logger.info(f"Performing LLM literature search on top {top_n} predictions...")

        llm = LLMLiteratureSearch(
            provider=self.config['methods']['llm_validation'].get('provider', 'openai'),
            model=self.config['methods']['llm_validation'].get('model', 'gpt-4')
        )

        # Prioritize interactions for LLM validation
        prioritized = llm.prioritize_for_validation(predictions, top_n=top_n)

        # Validate with LLM
        validated = llm.batch_validate_interactions(
            prioritized,
            max_workers=self.config['methods']['llm_validation'].get('parallel_requests', 5)
        )

        # Merge validated results back
        validated_map = {(v['protein_a'], v['protein_b']): v for v in validated}

        final_predictions = []
        for pred in predictions:
            key = (pred['protein_a'], pred['protein_b'])
            if key in validated_map:
                final_predictions.append(validated_map[key])
            else:
                final_predictions.append(pred)

        logger.info("LLM literature search complete")
        return final_predictions

    def generate_report(self, predictions: List[Dict], metadata: Dict) -> str:
        """
        Generate comprehensive report

        Args:
            predictions: Final predictions with all enrichments
            metadata: Analysis metadata

        Returns:
            Path to generated report
        """
        logger.info("Generating report...")

        generator = ReportGenerator(
            output_dir=self.config.get('data', {}).get('output_dir', './results')
        )

        report_path = generator.generate_full_report(predictions, metadata)
        csv_path = generator.generate_csv_export(predictions)

        logger.info(f"Report generated: {report_path}")
        logger.info(f"CSV exported: {csv_path}")

        return report_path

    def run(self, input_file: str, output_dir: str = "./results",
            threshold: float = 0.5, llm_top_n: int = 20):
        """
        Run complete AutoCoEv v2 analysis pipeline

        Args:
            input_file: Input FASTA file
            output_dir: Output directory for results
            threshold: Minimum score threshold
            llm_top_n: Number of top predictions to validate with LLM

        Returns:
            Path to generated report
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("AutoCoEv v2 Modern Pipeline")
        logger.info("=" * 60)

        # Load proteins
        proteins = self.load_proteins(input_file)

        # Generate pairs
        pairs = self.generate_pairs(proteins)

        # Analyze with ESM-2
        predictions = self.analyze_pairs(pairs, threshold=threshold)

        # Validate with STRING
        predictions = self.validate_with_string(predictions)

        # Literature search with LLM
        predictions = self.literature_search(predictions, top_n=llm_top_n)

        # Generate report
        runtime = time.time() - start_time
        metadata = {
            'method': 'ESM-2 + STRING + LLM',
            'n_proteins': len(proteins),
            'n_pairs': len(pairs),
            'runtime': f"{runtime/60:.1f} minutes",
            'threshold': threshold
        }

        report_path = self.generate_report(predictions, metadata)

        logger.info("=" * 60)
        logger.info(f"Analysis complete in {runtime/60:.1f} minutes")
        logger.info(f"Results saved to: {report_path}")
        logger.info("=" * 60)

        return report_path


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="AutoCoEv v2 - Modern protein-protein interaction prediction"
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help="Input FASTA file with protein sequences"
    )

    parser.add_argument(
        '--output', '-o',
        default='./results',
        help="Output directory for results (default: ./results)"
    )

    parser.add_argument(
        '--config', '-c',
        help="Configuration file (default: modern/config/config.yaml)"
    )

    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help="Minimum score threshold (default: 0.5)"
    )

    parser.add_argument(
        '--llm-top',
        type=int,
        default=20,
        help="Number of top predictions to validate with LLM (default: 20)"
    )

    parser.add_argument(
        '--no-string',
        action='store_true',
        help="Disable STRING database validation"
    )

    parser.add_argument(
        '--no-llm',
        action='store_true',
        help="Disable LLM literature search"
    )

    args = parser.parse_args()

    # Initialize pipeline
    config_path = args.config or "modern/config/config.yaml"
    pipeline = AutoCoEvModern(config_path=config_path)

    # Override config based on arguments
    if args.no_string:
        pipeline.config['methods']['string_db']['enabled'] = False

    if args.no_llm:
        pipeline.config['methods']['llm_validation']['enabled'] = False

    # Run analysis
    try:
        report_path = pipeline.run(
            input_file=args.input,
            output_dir=args.output,
            threshold=args.threshold,
            llm_top_n=args.llm_top
        )

        print(f"\n✓ Analysis complete!")
        print(f"✓ Report: {report_path}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
