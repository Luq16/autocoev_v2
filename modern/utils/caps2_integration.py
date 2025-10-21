"""
CAPS2 Integration Utilities
Converts ESM-2 results to CAPS2 format and prepares CAPS2 execution
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_top_candidates(esm2_results_csv: str,
                           top_n: int = 20,
                           min_confidence: float = 0.6,
                           prioritize_novel: bool = True) -> pd.DataFrame:
    """
    Extract top N candidates from ESM-2 results for CAPS2 validation

    Args:
        esm2_results_csv: Path to ESM-2 results CSV
        top_n: Number of top candidates to select
        min_confidence: Minimum combined confidence threshold
        prioritize_novel: Give priority to novel discoveries

    Returns:
        DataFrame with top candidates
    """
    logger.info(f"Extracting top {top_n} candidates from ESM-2 results...")

    # Load ESM-2 results
    df = pd.read_csv(esm2_results_csv)

    # Filter by minimum confidence
    df_filtered = df[df['combined_confidence'] >= min_confidence].copy()

    logger.info(f"Found {len(df_filtered)} interactions above confidence {min_confidence}")

    # Prioritization strategy
    if prioritize_novel:
        # Split into novel and known
        novel = df_filtered[df_filtered['is_novel'] == True].copy()
        known = df_filtered[df_filtered['is_novel'] == False].copy()

        # Sort each by confidence
        novel = novel.sort_values('combined_confidence', ascending=False)
        known = known.sort_values('combined_confidence', ascending=False)

        # Take top 70% from novel, 30% from known (if available)
        novel_count = min(len(novel), int(top_n * 0.7))
        known_count = min(len(known), top_n - novel_count)

        # Combine
        top_candidates = pd.concat([
            novel.head(novel_count),
            known.head(known_count)
        ]).sort_values('combined_confidence', ascending=False)

        logger.info(f"Selected {novel_count} novel + {known_count} known interactions")
    else:
        # Simple top-N by confidence
        top_candidates = df_filtered.sort_values('combined_confidence', ascending=False).head(top_n)

    logger.info(f"Top candidate: {top_candidates.iloc[0]['protein_a']} - {top_candidates.iloc[0]['protein_b']} (confidence: {top_candidates.iloc[0]['combined_confidence']:.3f})")

    return top_candidates


def convert_to_caps2_format(top_candidates: pd.DataFrame,
                            protein_sequences: Dict[str, str],
                            output_dir: str) -> Tuple[str, List[str]]:
    """
    Convert top candidates to CAPS2-compatible FASTA format

    Args:
        top_candidates: DataFrame with top protein pairs
        protein_sequences: Dict mapping protein_id to sequence
        output_dir: Output directory for FASTA files

    Returns:
        Tuple of (fasta_path, list of unique protein IDs)
    """
    logger.info("Converting candidates to CAPS2 FASTA format...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get unique proteins from candidates
    unique_proteins = set()
    for _, row in top_candidates.iterrows():
        unique_proteins.add(row['protein_a'])
        unique_proteins.add(row['protein_b'])

    unique_proteins = sorted(list(unique_proteins))
    logger.info(f"Unique proteins in top candidates: {len(unique_proteins)}")

    # Write FASTA file
    fasta_path = output_path / "caps2_input_proteins.fasta"

    with open(fasta_path, 'w') as f:
        for protein_id in unique_proteins:
            if protein_id in protein_sequences:
                sequence = protein_sequences[protein_id]
                f.write(f">{protein_id}\n")
                # Write sequence in 60-character lines (standard FASTA format)
                for i in range(0, len(sequence), 60):
                    f.write(sequence[i:i+60] + "\n")
            else:
                logger.warning(f"Sequence not found for {protein_id}")

    logger.info(f"FASTA file created: {fasta_path}")

    return str(fasta_path), unique_proteins


def prepare_protein_list(unique_proteins: List[str],
                         output_dir: str,
                         list_name: str = "tier2_candidates") -> str:
    """
    Prepare protein list file for CAPS2

    Args:
        unique_proteins: List of protein IDs
        output_dir: Output directory
        list_name: Name for the protein list file

    Returns:
        Path to protein list file
    """
    logger.info("Preparing CAPS2 protein list...")

    output_path = Path(output_dir)
    proteins_dir = output_path / "proteins"
    proteins_dir.mkdir(parents=True, exist_ok=True)

    list_path = proteins_dir / f"{list_name}.tsv"

    # Write protein list (one per line)
    with open(list_path, 'w') as f:
        for protein_id in unique_proteins:
            f.write(f"{protein_id}\n")

    logger.info(f"Protein list created: {list_path} ({len(unique_proteins)} proteins)")

    return str(list_path)


def generate_caps2_config(output_dir: str,
                          working_dir: str,
                          database_dir: str,
                          organism: str = "10090",
                          level: str = "40674",
                          protein_list: str = "tier2_candidates.tsv",
                          species_list: str = "placental.tsv",
                          template_config: str = None) -> str:
    """
    Generate CAPS2 settings.conf for two-tier analysis

    Args:
        output_dir: Output directory for config
        working_dir: CAPS2 working directory
        database_dir: Database directory path
        organism: Taxid of reference organism (default: 10090 = Mouse)
        level: Taxonomic level for ortholog search (default: 40674 = Mammalia)
        protein_list: Name of protein list file
        species_list: Name of species list file
        template_config: Path to template config (if None, use defaults)

    Returns:
        Path to generated settings.conf
    """
    logger.info("Generating CAPS2 configuration...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / "settings.conf"

    # Configuration template optimized for two-tier workflow
    config_content = f"""#!/bin/bash

## AutoCoEv CONFIGURATION FILE (Two-Tier Workflow)
## Auto-generated for CAPS2 validation of ESM-2 top candidates

## INPUT FILES
PROTEIN="proteins/"     # Folder with protein list
SPECIES="{species_list}"   # Species list
EXTTREE="placental.nwk"   # External species tree file
PAIRLST=""		# Protein pairs (empty = all pairs)

## REFERENCE ORGANISM AND ORTHOLOGUES
ORGANISM="{organism}"	# Reference organism taxid
LEVEL="{level}"		# Taxonomic level for orthologs

## WORKING AND DATABASE DIRS
TMP="{working_dir}"	# Working folder
DTB="{database_dir}"	# Database folder

## THREADS UTILIZATION
THREADS="$(nproc)"	# Auto-detect cores

## BLAST OPTIONS
DETBLAST="yes"		# Detailed BLAST results
PIDENT="35.000"		# Minimum identity (%)
PGAPS="25"		# Maximum gaps (%)
BLASTCORES="4"		# BLAST threads

# ORTHOLOGUES ASSESSMENT BY GUIDANCE
GUIDANCEMSA="muscle"	# MSA method for GUIDANCE
GUIDANCECUT="0.95"	# Sequence cutoff value
GUIDANCEKEEP="keep"	# Keep additional results

## MSA OPTIONS
MSAMETHOD="prank"	# MSA method (prank recommended for CAPS)
MUSCLEOPTIONS=""	# MUSCLE options
MAFFTOPTIONS=""		# MAFFT options
PRANKOPTIONS=""		# PRANK options
PRANKGUIDE="noguide"	# Use external guide tree
GBLOCKSOPT="-b5=h"	# Gblocks options (half gaps)

## PhyML OPTIONS
PHYMLOPTIONS=""		# PhyML options
PHYMLGUIDE="noguide"	# Use external guide tree
TREESROOT="rooted"	# Root trees with TreeBeST
TREESCAPS="auto"	# Let CAPS generate trees

## PAIRING
PAIRINGMANNER="all"	# Pairing manner (all pairs)
MINCOMMONSPCS="20"	# Minimum common species per pair
INCR="1000"		# Divide pairs into groups

## CAPS RUN-TIME OPTIONS
ALPHA="0.01"		# Alpha value for threshold
BOOT="0.6"		# Bootstrap threshold
CAPSOPTIONS="-c"	# CAPS options (convergence)
REFER="-H ${{ORGANISM}}"	# Reference organism
PVALUE="$ALPHA"		# Post-run P-value cutoff

### DATABASES SECTION ###

## DATABASES version
ORTHODBVER="v101"	# OrthoDB version

## Databases (names only, no extensions)
GENEXREFALL="odb10v1_gene_xrefs"
OG2GENESALL="odb10v1_OG2genes"
ALLFASTA="odb10v1_all_fasta"

## MD5SUMs of databases (gzipped)
GENEXREFALLM5="3ab6d2efdc43ed051591514a3cc9044e"
OG2GENESALLM5="33e63fa97ee420707cd3cddcb5e282a6"
ALLFASTAM5="831ef830fff549857a4c8d1639a760cb"
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    logger.info(f"CAPS2 configuration created: {config_path}")

    return str(config_path)


def merge_results(esm2_df: pd.DataFrame,
                 caps2_results: Dict[Tuple[str, str], Dict],
                 output_path: str) -> pd.DataFrame:
    """
    Merge ESM-2 and CAPS2 results into unified dataset

    Args:
        esm2_df: DataFrame with ESM-2 results
        caps2_results: Dict mapping (protein_a, protein_b) to CAPS2 scores
        output_path: Path to save merged results

    Returns:
        Merged DataFrame
    """
    logger.info("Merging ESM-2 and CAPS2 results...")

    # Add CAPS2 columns
    merged_df = esm2_df.copy()
    merged_df['caps2_score'] = 0.0
    merged_df['caps2_validated'] = False
    merged_df['caps2_pvalue'] = 1.0
    merged_df['caps2_coev_count'] = 0
    merged_df['validation_method'] = 'ESM-2 only'

    # Merge CAPS2 results
    validated_count = 0
    for idx, row in merged_df.iterrows():
        pair_key = (row['protein_a'], row['protein_b'])
        reverse_key = (row['protein_b'], row['protein_a'])

        if pair_key in caps2_results:
            caps2_data = caps2_results[pair_key]
            merged_df.at[idx, 'caps2_score'] = caps2_data.get('score', 0.0)
            merged_df.at[idx, 'caps2_validated'] = True
            merged_df.at[idx, 'caps2_pvalue'] = caps2_data.get('pvalue', 1.0)
            merged_df.at[idx, 'caps2_coev_count'] = caps2_data.get('coev_count', 0)
            merged_df.at[idx, 'validation_method'] = 'ESM-2 + CAPS2'
            validated_count += 1
        elif reverse_key in caps2_results:
            caps2_data = caps2_results[reverse_key]
            merged_df.at[idx, 'caps2_score'] = caps2_data.get('score', 0.0)
            merged_df.at[idx, 'caps2_validated'] = True
            merged_df.at[idx, 'caps2_pvalue'] = caps2_data.get('pvalue', 1.0)
            merged_df.at[idx, 'caps2_coev_count'] = caps2_data.get('coev_count', 0)
            merged_df.at[idx, 'validation_method'] = 'ESM-2 + CAPS2'
            validated_count += 1

    logger.info(f"Validated {validated_count} interactions with CAPS2")

    # Recalculate combined confidence (ESM-2 + CAPS2 + STRING)
    merged_df['final_confidence'] = merged_df.apply(
        lambda row: calculate_final_confidence(row), axis=1
    )

    # Sort by final confidence
    merged_df = merged_df.sort_values('final_confidence', ascending=False)

    # Save merged results
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Merged results saved: {output_path}")

    return merged_df


def calculate_final_confidence(row: pd.Series) -> float:
    """
    Calculate final confidence score combining ESM-2, CAPS2, and STRING

    Args:
        row: DataFrame row with scores

    Returns:
        Final confidence score (0-1)
    """
    esm2_score = row.get('autocoev_score', row.get('score', 0.0))
    caps2_score = row.get('caps2_score', 0.0)
    string_score = row.get('string_score', 0.0)
    caps2_validated = row.get('caps2_validated', False)

    # Weighted average based on available data
    if caps2_validated:
        # All three methods available
        weights = {'esm2': 0.35, 'caps2': 0.40, 'string': 0.25}
        final = (
            esm2_score * weights['esm2'] +
            caps2_score * weights['caps2'] +
            string_score * weights['string']
        )
    else:
        # Only ESM-2 + STRING (original combined_confidence)
        final = row.get('combined_confidence', esm2_score * 0.6 + string_score * 0.4)

    return final


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python caps2_integration.py <esm2_results.csv>")
        sys.exit(1)

    # Extract top candidates
    top_candidates = extract_top_candidates(
        esm2_results_csv=sys.argv[1],
        top_n=20,
        prioritize_novel=True
    )

    print(f"\nTop {len(top_candidates)} candidates:")
    print(top_candidates[['protein_a', 'protein_b', 'combined_confidence', 'is_novel']])
