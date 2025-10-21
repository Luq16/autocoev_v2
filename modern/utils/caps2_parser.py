"""
CAPS2 Results Parser
Parses CAPS2 output files (allResidues format) and extracts coevolution data
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CAPS2ResultsParser:
    """
    Parser for CAPS2 output files
    """

    def __init__(self, results_dir: str):
        """
        Initialize CAPS2 results parser

        Args:
            results_dir: Directory containing CAPS2 results
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            logger.warning(f"Results directory does not exist: {results_dir}")

    def find_result_files(self, pattern: str = "*/*/allResidues") -> List[Path]:
        """
        Find all CAPS2 result files

        Args:
            pattern: Glob pattern for result files

        Returns:
            List of paths to result files
        """
        result_files = list(self.results_dir.glob(pattern))
        logger.info(f"Found {len(result_files)} CAPS2 result files")
        return result_files

    def parse_all_residues_file(self, file_path: Path) -> Dict:
        """
        Parse a single allResidues file

        Args:
            file_path: Path to allResidues file

        Returns:
            Dictionary with parsed coevolution data
        """
        try:
            # Extract protein pair from path
            # Path format: results_dir/{fold}/{proteinA_proteinB}/allResidues
            pair_dir = file_path.parent.name
            protein_a, protein_b = self.extract_protein_pair(pair_dir)

            # Read file
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Parse coevolving residues
            coev_residues = []
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        try:
                            residue_data = {
                                'pos_a': int(parts[0]),  # Position in protein A
                                'aa_a': parts[1],        # Amino acid in protein A
                                'pos_b': int(parts[2]),  # Position in protein B
                                'aa_b': parts[3],        # Amino acid in protein B
                                'pvalue': float(parts[4]) if parts[4] != 'NA' else 1.0,
                                'score': float(parts[5]) if parts[5] != 'NA' else 0.0
                            }
                            coev_residues.append(residue_data)
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Skipping malformed line: {line.strip()}")
                            continue

            # Calculate aggregate statistics
            if coev_residues:
                pvalues = [r['pvalue'] for r in coev_residues if r['pvalue'] < 1.0]
                scores = [r['score'] for r in coev_residues if r['score'] > 0]

                avg_pvalue = sum(pvalues) / len(pvalues) if pvalues else 1.0
                avg_score = sum(scores) / len(scores) if scores else 0.0
                max_score = max(scores) if scores else 0.0
            else:
                avg_pvalue = 1.0
                avg_score = 0.0
                max_score = 0.0

            result = {
                'protein_a': protein_a,
                'protein_b': protein_b,
                'coev_count': len(coev_residues),
                'pvalue': avg_pvalue,
                'score': max_score,  # Use max score as representative
                'avg_score': avg_score,
                'residues': coev_residues
            }

            logger.debug(f"Parsed {protein_a}-{protein_b}: {len(coev_residues)} coevolving residues")

            return result

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def extract_protein_pair(self, pair_dir_name: str) -> Tuple[str, str]:
        """
        Extract protein pair from directory name

        Args:
            pair_dir_name: Directory name (e.g., "PROTEIN1_PROTEIN2")

        Returns:
            Tuple of (protein_a, protein_b)
        """
        # Handle various naming conventions
        # Common patterns: "PROT1_PROT2", "PROT1-PROT2", "PROT1.PROT2"
        separators = ['_', '-', '.']

        for sep in separators:
            if sep in pair_dir_name:
                parts = pair_dir_name.split(sep, 1)
                if len(parts) == 2:
                    return parts[0], parts[1]

        logger.warning(f"Could not extract protein pair from: {pair_dir_name}")
        return pair_dir_name, ""

    def parse_all_results(self) -> Dict[Tuple[str, str], Dict]:
        """
        Parse all CAPS2 result files in the results directory

        Returns:
            Dictionary mapping (protein_a, protein_b) to result data
        """
        logger.info("Parsing all CAPS2 results...")

        result_files = self.find_result_files()
        all_results = {}

        for file_path in result_files:
            result = self.parse_all_residues_file(file_path)
            if result:
                pair_key = (result['protein_a'], result['protein_b'])
                all_results[pair_key] = result

        logger.info(f"Successfully parsed {len(all_results)} protein pairs")

        return all_results


def parse_all_residues(results_dir: str) -> Dict[Tuple[str, str], Dict]:
    """
    Convenience function to parse all CAPS2 results

    Args:
        results_dir: Directory containing CAPS2 results

    Returns:
        Dictionary mapping protein pairs to coevolution data
    """
    parser = CAPS2ResultsParser(results_dir)
    return parser.parse_all_results()


def merge_caps2_with_esm2(esm2_csv: str,
                          caps2_results_dir: str,
                          output_csv: str) -> pd.DataFrame:
    """
    Merge CAPS2 results with ESM-2 results

    Args:
        esm2_csv: Path to ESM-2 results CSV
        caps2_results_dir: Directory with CAPS2 results
        output_csv: Path to save merged results

    Returns:
        Merged DataFrame
    """
    logger.info("Merging CAPS2 and ESM-2 results...")

    # Load ESM-2 results
    esm2_df = pd.read_csv(esm2_csv)

    # Parse CAPS2 results
    caps2_results = parse_all_residues(caps2_results_dir)

    # Add CAPS2 columns
    esm2_df['caps2_score'] = 0.0
    esm2_df['caps2_validated'] = False
    esm2_df['caps2_pvalue'] = 1.0
    esm2_df['caps2_coev_count'] = 0

    # Merge CAPS2 data
    for idx, row in esm2_df.iterrows():
        pair_key = (row['protein_a'], row['protein_b'])
        reverse_key = (row['protein_b'], row['protein_a'])

        caps2_data = caps2_results.get(pair_key) or caps2_results.get(reverse_key)

        if caps2_data:
            esm2_df.at[idx, 'caps2_score'] = caps2_data['score']
            esm2_df.at[idx, 'caps2_validated'] = True
            esm2_df.at[idx, 'caps2_pvalue'] = caps2_data['pvalue']
            esm2_df.at[idx, 'caps2_coev_count'] = caps2_data['coev_count']

    # Calculate final confidence
    esm2_df['final_confidence'] = esm2_df.apply(
        lambda row: calculate_weighted_confidence(row), axis=1
    )

    # Sort by final confidence
    merged_df = esm2_df.sort_values('final_confidence', ascending=False)

    # Save
    merged_df.to_csv(output_csv, index=False)
    logger.info(f"Merged results saved: {output_csv}")

    # Summary statistics
    validated_count = merged_df['caps2_validated'].sum()
    logger.info(f"CAPS2 validated {validated_count}/{len(merged_df)} interactions")

    return merged_df


def calculate_weighted_confidence(row: pd.Series) -> float:
    """
    Calculate weighted confidence combining all methods

    Args:
        row: DataFrame row

    Returns:
        Weighted confidence score
    """
    esm2_score = row.get('autocoev_score', row.get('score', 0.0))
    string_score = row.get('string_score', 0.0)
    caps2_score = row.get('caps2_score', 0.0)
    caps2_validated = row.get('caps2_validated', False)

    if caps2_validated:
        # Three-way fusion: ESM-2 (35%) + CAPS2 (40%) + STRING (25%)
        weights = {'esm2': 0.35, 'caps2': 0.40, 'string': 0.25}
        confidence = (
            esm2_score * weights['esm2'] +
            caps2_score * weights['caps2'] +
            string_score * weights['string']
        )
    else:
        # Two-way fusion: ESM-2 (60%) + STRING (40%)
        confidence = row.get('combined_confidence', esm2_score * 0.6 + string_score * 0.4)

    return confidence


def export_residue_level_data(caps2_results: Dict[Tuple[str, str], Dict],
                              output_dir: str):
    """
    Export residue-level coevolution data to CSV files

    Args:
        caps2_results: Dictionary of CAPS2 results
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for (protein_a, protein_b), data in caps2_results.items():
        if data['residues']:
            residue_file = output_path / f"{protein_a}_{protein_b}_residues.csv"

            residue_df = pd.DataFrame(data['residues'])
            residue_df.to_csv(residue_file, index=False)

            logger.debug(f"Exported residue data: {residue_file}")

    logger.info(f"Exported residue-level data for {len(caps2_results)} pairs")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python caps2_parser.py <caps2_results_dir>")
        sys.exit(1)

    # Parse CAPS2 results
    results = parse_all_residues(sys.argv[1])

    print(f"\nParsed {len(results)} protein pairs:")
    for (prot_a, prot_b), data in list(results.items())[:5]:
        print(f"{prot_a} - {prot_b}: {data['coev_count']} coevolving residues, score: {data['score']:.4f}")
