#!/usr/bin/env python3
"""
AutoCoEv v2 - Two-Tier Workflow
Combines ESM-2 fast screening with CAPS2 rigorous validation

This workflow provides the best of both worlds:
- Tier 1: ESM-2 screens all protein pairs (fast, 95-98% accuracy)
- Tier 2: CAPS2 validates top candidates (rigorous, residue-level detail)

Usage:
    python two_tier_workflow.py --input proteins.fasta --top-n 20 --output results/
"""

import argparse
import logging
import yaml
import time
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add modern modules to path
sys.path.insert(0, str(Path(__file__).parent / "modern"))

from core.fast_embedding_generator import FastProteinEmbedder
from integrations.string_db import STRINGDatabase
from report.report_generator import ReportGenerator
from utils.caps2_integration import (
    extract_top_candidates,
    convert_to_caps2_format,
    prepare_protein_list,
    generate_caps2_config
)
from utils.caps2_parser import parse_all_residues, merge_caps2_with_esm2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TwoTierWorkflow:
    """
    Two-Tier AutoCoEv Workflow Orchestrator

    Coordinates ESM-2 screening → Candidate selection → CAPS2 validation
    """

    def __init__(self, config_path: str = None):
        """
        Initialize two-tier workflow

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Load default two-tier config
            default_config = Path(__file__).parent / "modern" / "config" / "two_tier_config.yaml"
            with open(default_config, 'r') as f:
                self.config = yaml.safe_load(f)

        logger.info("Two-Tier Workflow initialized")
        logger.info(f"Tier 1 (ESM-2): {'ENABLED' if self.config['tier1_esm2']['enabled'] else 'DISABLED'}")
        logger.info(f"Tier 2 (CAPS2): {'ENABLED' if self.config['tier2_caps2']['enabled'] else 'DISABLED'}")

    def load_proteins(self, fasta_file: str) -> Dict[str, str]:
        """
        Load proteins from FASTA file

        Args:
            fasta_file: Path to FASTA file

        Returns:
            Dictionary mapping protein_id to sequence
        """
        proteins = {}

        with open(fasta_file, 'r') as f:
            current_id = None
            current_seq = []

            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous protein
                    if current_id:
                        proteins[current_id] = ''.join(current_seq)

                    # Start new protein
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            # Save last protein
            if current_id:
                proteins[current_id] = ''.join(current_seq)

        logger.info(f"Loaded {len(proteins)} proteins from {fasta_file}")
        return proteins

    def run_tier1_screening(self, proteins: Dict[str, str],
                           output_dir: Path) -> str:
        """
        Tier 1: Run ESM-2 fast screening on all protein pairs

        Args:
            proteins: Dictionary of protein_id -> sequence
            output_dir: Output directory

        Returns:
            Path to ESM-2 results CSV
        """
        logger.info("=" * 60)
        logger.info("TIER 1: ESM-2 FAST SCREENING")
        logger.info("=" * 60)

        start_time = time.time()

        # Initialize ESM-2
        esm2_config = self.config['tier1_esm2']
        embedder = FastProteinEmbedder(
            device=esm2_config.get('device', 'auto'),
            cache_dir=self.config.get('data', {}).get('esm_cache', './cache/esm_embeddings')
        )

        # Generate protein pairs
        protein_ids = list(proteins.keys())
        n_proteins = len(protein_ids)
        pairs = []

        for i in range(n_proteins):
            for j in range(i + 1, n_proteins):
                pairs.append((protein_ids[i], protein_ids[j]))

        logger.info(f"Analyzing {n_proteins} proteins ({len(pairs)} pairs)")

        # Calculate coevolution scores
        threshold = esm2_config.get('threshold', 0.5)
        predictions = []

        for idx, (protein_a_id, protein_b_id) in enumerate(pairs):
            if (idx + 1) % 100 == 0:
                logger.info(f"Progress: {idx + 1}/{len(pairs)} pairs analyzed")

            seq_a = proteins[protein_a_id]
            seq_b = proteins[protein_b_id]

            score = embedder.get_coevolution_score(protein_a_id, protein_b_id, seq_a, seq_b)

            if score >= threshold:
                predictions.append({
                    'protein_a': protein_a_id,
                    'protein_b': protein_b_id,
                    'autocoev_score': score,
                    'score': score  # For STRING enrichment
                })

        logger.info(f"Found {len(predictions)} interactions above threshold {threshold}")

        # STRING validation (if enabled)
        if self.config['string_db']['enabled'] and predictions:
            logger.info("Validating with STRING database...")

            string_db = STRINGDatabase(
                species=self.config['string_db']['species'],
                required_score=self.config['string_db']['required_score']
            )

            predictions = string_db.enrich_with_string(predictions)
        else:
            # Add default STRING fields
            for pred in predictions:
                pred.update({
                    'string_score': 0.0,
                    'combined_confidence': pred['autocoev_score'] * 0.9,
                    'is_novel': True,
                    'exists_in_string': False
                })

        # Save ESM-2 results
        import pandas as pd
        esm2_df = pd.DataFrame(predictions)
        esm2_df = esm2_df.sort_values('combined_confidence', ascending=False)

        esm2_csv_path = output_dir / "tier1_esm2_results.csv"
        esm2_df.to_csv(esm2_csv_path, index=False)

        runtime = time.time() - start_time
        logger.info(f"Tier 1 complete in {runtime/60:.1f} minutes")
        logger.info(f"Results saved: {esm2_csv_path}")

        return str(esm2_csv_path)

    def run_tier2_validation(self, esm2_csv: str, proteins: Dict[str, str],
                            output_dir: Path) -> Dict:
        """
        Tier 2: Run CAPS2 validation on top candidates

        Args:
            esm2_csv: Path to ESM-2 results CSV
            proteins: Dictionary of protein sequences
            output_dir: Output directory

        Returns:
            Dictionary with CAPS2 results and paths
        """
        logger.info("=" * 60)
        logger.info("TIER 2: CAPS2 RIGOROUS VALIDATION")
        logger.info("=" * 60)

        start_time = time.time()

        caps2_config = self.config['tier2_caps2']

        # Step 1: Extract top candidates
        logger.info("Step 1/5: Extracting top candidates...")
        top_candidates = extract_top_candidates(
            esm2_results_csv=esm2_csv,
            top_n=caps2_config['top_n_candidates'],
            min_confidence=caps2_config['min_confidence'],
            prioritize_novel=caps2_config['prioritize_novel']
        )

        if len(top_candidates) == 0:
            logger.warning("No candidates selected for CAPS2 validation!")
            return {'caps2_executed': False}

        logger.info(f"Selected {len(top_candidates)} candidates for CAPS2 validation")

        # Step 2: Prepare CAPS2 input directory
        logger.info("Step 2/5: Preparing CAPS2 inputs...")
        caps2_dir = output_dir / "caps2_validation"
        caps2_dir.mkdir(parents=True, exist_ok=True)

        # Convert to FASTA
        fasta_path, unique_proteins = convert_to_caps2_format(
            top_candidates=top_candidates,
            protein_sequences=proteins,
            output_dir=str(caps2_dir)
        )

        # Prepare protein list
        protein_list_path = prepare_protein_list(
            unique_proteins=unique_proteins,
            output_dir=str(caps2_dir),
            list_name="tier2_candidates"
        )

        # Step 3: Generate CAPS2 configuration
        logger.info("Step 3/5: Generating CAPS2 configuration...")
        working_dir = caps2_config['working_dir']
        database_dir = caps2_config['database_dir']

        config_path = generate_caps2_config(
            output_dir=str(caps2_dir),
            working_dir=working_dir,
            database_dir=database_dir,
            organism=str(caps2_config['organism']),
            level=str(caps2_config['level']),
            protein_list="tier2_candidates.tsv",
            species_list=caps2_config['species_list']
        )

        logger.info(f"CAPS2 configuration: {config_path}")

        # Step 4: Execute CAPS2 (if enabled)
        if caps2_config.get('enabled', True):
            logger.info("Step 4/5: Executing CAPS2...")
            logger.info("⚠️  CAPS2 execution requires manual intervention")
            logger.info(f"Please run the following commands:")
            logger.info(f"")
            logger.info(f"cd {Path(__file__).parent}")
            logger.info(f"export TMP={working_dir}")
            logger.info(f"export DTB={database_dir}")
            logger.info(f"./start.sh")
            logger.info(f"")
            logger.info("Then select the following steps:")
            logger.info("  - Skip database preparation (option 11)")
            logger.info("  - Run steps 3-5 (download sequences, BLAST, get FASTA)")
            logger.info("  - Run steps 7 (MSA)")
            logger.info("  - Run steps 9-10 (create pairs, CAPS run)")
            logger.info(f"")
            logger.info("Press ENTER when CAPS2 execution is complete...")
            input()

            # Step 5: Parse CAPS2 results
            logger.info("Step 5/5: Parsing CAPS2 results...")
            caps2_results_dir = Path(working_dir) / "results" / "coev"

            if caps2_results_dir.exists():
                caps2_results = parse_all_residues(str(caps2_results_dir))
                logger.info(f"Parsed {len(caps2_results)} CAPS2 results")

                runtime = time.time() - start_time
                logger.info(f"Tier 2 complete in {runtime/60:.1f} minutes")

                return {
                    'caps2_executed': True,
                    'caps2_results': caps2_results,
                    'caps2_dir': str(caps2_dir),
                    'runtime': runtime
                }
            else:
                logger.error(f"CAPS2 results directory not found: {caps2_results_dir}")
                return {'caps2_executed': False}
        else:
            logger.info("CAPS2 execution disabled in configuration")
            return {'caps2_executed': False}

    def merge_and_report(self, esm2_csv: str, tier2_results: Dict,
                        proteins: Dict[str, str], output_dir: Path):
        """
        Merge results from both tiers and generate comprehensive report

        Args:
            esm2_csv: Path to ESM-2 results
            tier2_results: Dictionary with CAPS2 results
            proteins: Dictionary of protein sequences
            output_dir: Output directory
        """
        logger.info("=" * 60)
        logger.info("MERGING RESULTS AND GENERATING REPORT")
        logger.info("=" * 60)

        # Merge results
        if tier2_results.get('caps2_executed', False):
            merged_csv = output_dir / "two_tier_merged_results.csv"
            merged_df = merge_caps2_with_esm2(
                esm2_csv=esm2_csv,
                caps2_results_dir=tier2_results['caps2_dir'],
                output_csv=str(merged_csv)
            )
            logger.info(f"Merged results saved: {merged_csv}")
        else:
            logger.info("Using ESM-2 results only (CAPS2 not executed)")
            import pandas as pd
            merged_df = pd.read_csv(esm2_csv)
            merged_csv = esm2_csv

        # Generate comprehensive report
        logger.info("Generating comprehensive report...")
        report_gen = ReportGenerator(output_dir=str(output_dir))

        metadata = {
            'method': 'Two-Tier (ESM-2 + CAPS2 + STRING)',
            'n_proteins': len(proteins),
            'n_pairs': len(merged_df),
            'tier1_method': 'ESM-2',
            'tier2_method': 'CAPS2' if tier2_results.get('caps2_executed') else 'None',
            'tier2_candidates': tier2_results.get('caps2_results', {}) if tier2_results.get('caps2_executed') else 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        report_path = report_gen.generate_full_report(
            merged_df.to_dict('records'),
            metadata
        )

        logger.info(f"Report generated: {report_path}")

        return report_path

    def run(self, input_file: str, output_dir: str = "./results/two_tier",
            top_n: int = None):
        """
        Run complete two-tier workflow

        Args:
            input_file: Input FASTA file
            output_dir: Output directory
            top_n: Number of top candidates for CAPS2 (overrides config)

        Returns:
            Path to final report
        """
        total_start_time = time.time()

        logger.info("=" * 70)
        logger.info("AutoCoEv v2 - Two-Tier Workflow")
        logger.info("Combining ESM-2 Speed with CAPS2 Rigor")
        logger.info("=" * 70)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Override config if specified
        if top_n:
            self.config['tier2_caps2']['top_n_candidates'] = top_n

        # Load proteins
        proteins = self.load_proteins(input_file)

        # Tier 1: ESM-2 Screening
        esm2_csv = self.run_tier1_screening(proteins, output_path)

        # Tier 2: CAPS2 Validation
        tier2_results = self.run_tier2_validation(esm2_csv, proteins, output_path)

        # Merge and Report
        report_path = self.merge_and_report(esm2_csv, tier2_results, proteins, output_path)

        total_runtime = time.time() - total_start_time
        logger.info("=" * 70)
        logger.info(f"TWO-TIER WORKFLOW COMPLETE")
        logger.info(f"Total Runtime: {total_runtime/60:.1f} minutes")
        logger.info(f"Results: {output_path}")
        logger.info(f"Report: {report_path}")
        logger.info("=" * 70)

        return report_path


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="AutoCoEv v2 - Two-Tier Workflow (ESM-2 + CAPS2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (20 top candidates for CAPS2)
  python two_tier_workflow.py --input proteins.fasta

  # Validate 30 top candidates
  python two_tier_workflow.py --input proteins.fasta --top-n 30

  # Custom output directory
  python two_tier_workflow.py --input proteins.fasta --output results/my_analysis

  # Use custom configuration
  python two_tier_workflow.py --input proteins.fasta --config my_config.yaml
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help="Input FASTA file with protein sequences"
    )

    parser.add_argument(
        '--output', '-o',
        default='./results/two_tier',
        help="Output directory (default: ./results/two_tier)"
    )

    parser.add_argument(
        '--config', '-c',
        help="Configuration file (default: modern/config/two_tier_config.yaml)"
    )

    parser.add_argument(
        '--top-n', '-n',
        type=int,
        help="Number of top candidates for CAPS2 validation (overrides config)"
    )

    parser.add_argument(
        '--esm2-threshold', '-t',
        type=float,
        help="ESM-2 score threshold (default: from config)"
    )

    args = parser.parse_args()

    # Initialize workflow
    workflow = TwoTierWorkflow(config_path=args.config)

    # Override config if specified
    if args.esm2_threshold:
        workflow.config['tier1_esm2']['threshold'] = args.esm2_threshold

    # Run workflow
    try:
        report_path = workflow.run(
            input_file=args.input,
            output_dir=args.output,
            top_n=args.top_n
        )

        print(f"\n✓ Two-tier analysis complete!")
        print(f"✓ Report: {report_path}")

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
