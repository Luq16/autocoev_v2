"""
Utility modules for AutoCoEv v2 pipeline integration
"""

from .caps2_integration import (
    extract_top_candidates,
    convert_to_caps2_format,
    generate_caps2_config,
    prepare_protein_list
)

from .caps2_parser import (
    CAPS2ResultsParser,
    parse_all_residues,
    merge_caps2_with_esm2
)

__all__ = [
    'extract_top_candidates',
    'convert_to_caps2_format',
    'generate_caps2_config',
    'prepare_protein_list',
    'CAPS2ResultsParser',
    'parse_all_residues',
    'merge_caps2_with_esm2'
]
