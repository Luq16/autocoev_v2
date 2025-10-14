"""
STRING Database Integration
Automatically query STRING database to validate predicted protein-protein interactions
"""

import requests
import logging
from typing import List, Dict, Optional, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STRINGDatabase:
    """
    Interface to STRING database for protein-protein interaction validation

    STRING provides known and predicted protein interactions with confidence scores
    """

    BASE_URL = "https://string-db.org/api"
    API_VERSION = "json"

    def __init__(self, species: int = 9606, required_score: int = 400):
        """
        Initialize STRING database client

        Args:
            species: NCBI taxonomy ID (9606 = Human, 10090 = Mouse, 7227 = Fly)
            required_score: Minimum interaction score threshold (0-1000)
                          400 = medium confidence
                          700 = high confidence
                          900 = highest confidence
        """
        self.species = species
        self.required_score = required_score

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def _rate_limit(self):
        """Ensure we don't overwhelm the STRING API"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def get_interaction(self, protein_a: str, protein_b: str) -> Optional[Dict]:
        """
        Query STRING for interaction between two proteins

        Args:
            protein_a: First protein identifier (gene name or UniProt ID)
            protein_b: Second protein identifier

        Returns:
            Dictionary with interaction data or None if not found:
            {
                'protein_a': str,
                'protein_b': str,
                'combined_score': int (0-1000),
                'experimental_score': int,
                'database_score': int,
                'coexpression_score': int,
                'neighborhood_score': int,
                'fusion_score': int,
                'cooccurrence_score': int,
                'textmining_score': int,
                'exists_in_string': bool
            }
        """
        self._rate_limit()

        # STRING network endpoint
        url = f"{self.BASE_URL}/{self.API_VERSION}/network"

        params = {
            'identifiers': f"{protein_a}%0d{protein_b}",  # %0d is newline separator
            'species': self.species,
            'required_score': self.required_score
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.debug(f"No interaction found in STRING for {protein_a}-{protein_b}")
                return {
                    'protein_a': protein_a,
                    'protein_b': protein_b,
                    'combined_score': 0,
                    'exists_in_string': False
                }

            # Parse first result (usually only one for direct interaction)
            interaction = data[0]

            return {
                'protein_a': protein_a,
                'protein_b': protein_b,
                'combined_score': int(interaction.get('score', 0) * 1000),  # Convert 0-1 to 0-1000
                'experimental_score': int(interaction.get('escore', 0) * 1000),
                'database_score': int(interaction.get('dscore', 0) * 1000),
                'coexpression_score': int(interaction.get('ascore', 0) * 1000),
                'neighborhood_score': int(interaction.get('nscore', 0) * 1000),
                'fusion_score': int(interaction.get('fscore', 0) * 1000),
                'cooccurrence_score': int(interaction.get('pscore', 0) * 1000),
                'textmining_score': int(interaction.get('tscore', 0) * 1000),
                'exists_in_string': True
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"STRING API error for {protein_a}-{protein_b}: {e}")
            return None

    def batch_get_interactions(self, protein_pairs: List[Tuple[str, str]],
                               parallel: bool = False) -> Dict[Tuple[str, str], Dict]:
        """
        Query STRING for multiple protein pairs

        Args:
            protein_pairs: List of (protein_a, protein_b) tuples
            parallel: Whether to use parallel requests (requires aiohttp)

        Returns:
            Dictionary mapping (protein_a, protein_b) to interaction data
        """
        results = {}

        logger.info(f"Querying STRING for {len(protein_pairs)} protein pairs...")

        for i, (protein_a, protein_b) in enumerate(protein_pairs):
            result = self.get_interaction(protein_a, protein_b)
            if result:
                results[(protein_a, protein_b)] = result

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(protein_pairs)} pairs")

        logger.info(f"STRING query complete: {sum(1 for r in results.values() if r.get('exists_in_string'))} found")

        return results

    def get_interaction_partners(self, protein: str, limit: int = 10) -> List[Dict]:
        """
        Get all known interaction partners for a protein from STRING

        Args:
            protein: Protein identifier
            limit: Maximum number of partners to return

        Returns:
            List of interaction partners with scores
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/{self.API_VERSION}/interaction_partners"

        params = {
            'identifiers': protein,
            'species': self.species,
            'required_score': self.required_score,
            'limit': limit
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            partners = []
            for item in data:
                partners.append({
                    'partner': item.get('preferredName_B', ''),
                    'score': int(item.get('score', 0) * 1000)
                })

            return partners

        except requests.exceptions.RequestException as e:
            logger.error(f"STRING API error for {protein}: {e}")
            return []

    def enrich_with_string(self, predictions: List[Dict]) -> List[Dict]:
        """
        Enrich AutoCoEv predictions with STRING database information

        Args:
            predictions: List of prediction dictionaries with 'protein_a', 'protein_b', 'score'

        Returns:
            Enriched predictions with STRING data and novelty flags
        """
        enriched = []

        logger.info(f"Enriching {len(predictions)} predictions with STRING data...")

        for pred in predictions:
            protein_a = pred['protein_a']
            protein_b = pred['protein_b']

            # Query STRING
            string_data = self.get_interaction(protein_a, protein_b)

            if string_data:
                # Combine AutoCoEv and STRING scores
                autocoev_score = pred.get('score', 0)
                string_score = string_data.get('combined_score', 0) / 1000  # Normalize to 0-1

                # Determine if this is a novel prediction
                is_novel = not string_data.get('exists_in_string', False)

                enriched_pred = {
                    **pred,
                    'string_score': string_score,
                    'string_experimental': string_data.get('experimental_score', 0) / 1000,
                    'string_database': string_data.get('database_score', 0) / 1000,
                    'string_textmining': string_data.get('textmining_score', 0) / 1000,
                    'exists_in_string': string_data.get('exists_in_string', False),
                    'is_novel': is_novel,
                    'combined_confidence': self._calculate_combined_confidence(
                        autocoev_score, string_score, is_novel
                    )
                }

                enriched.append(enriched_pred)

        logger.info(f"Enrichment complete. Novel predictions: {sum(1 for e in enriched if e['is_novel'])}")

        return enriched

    def _calculate_combined_confidence(self, autocoev_score: float,
                                      string_score: float, is_novel: bool) -> float:
        """
        Calculate combined confidence score

        For known interactions (in STRING):
            - High STRING score + High AutoCoEv = Very confident
            - High STRING score + Low AutoCoEv = STRING supported
            - Low STRING score + High AutoCoEv = Potentially interesting

        For novel interactions (not in STRING):
            - High AutoCoEv score = Novel discovery candidate

        Args:
            autocoev_score: Score from AutoCoEv analysis (0-1)
            string_score: Score from STRING database (0-1)
            is_novel: Whether this is a novel prediction

        Returns:
            Combined confidence score (0-1)
        """
        if is_novel:
            # For novel predictions, rely on AutoCoEv score
            # but be slightly more conservative
            return autocoev_score * 0.9

        else:
            # For known interactions, weighted average
            # STRING gets higher weight as it's experimental/curated
            return 0.4 * autocoev_score + 0.6 * string_score

    def get_functional_enrichment(self, proteins: List[str]) -> Dict:
        """
        Get functional enrichment analysis for a list of proteins

        Args:
            proteins: List of protein identifiers

        Returns:
            Functional enrichment data (GO terms, pathways, etc.)
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/{self.API_VERSION}/enrichment"

        params = {
            'identifiers': '%0d'.join(proteins),
            'species': self.species
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            enrichment = {
                'go_biological_process': [],
                'go_molecular_function': [],
                'go_cellular_component': [],
                'kegg_pathways': []
            }

            for item in data[:20]:  # Top 20 enriched terms
                category = item.get('category', '')
                if 'Biological Process' in category:
                    enrichment['go_biological_process'].append({
                        'term': item.get('term', ''),
                        'pvalue': item.get('p_value', 1.0),
                        'fdr': item.get('fdr', 1.0)
                    })
                elif 'Molecular Function' in category:
                    enrichment['go_molecular_function'].append({
                        'term': item.get('term', ''),
                        'pvalue': item.get('p_value', 1.0),
                        'fdr': item.get('fdr', 1.0)
                    })
                elif 'Cellular Component' in category:
                    enrichment['go_cellular_component'].append({
                        'term': item.get('term', ''),
                        'pvalue': item.get('p_value', 1.0),
                        'fdr': item.get('fdr', 1.0)
                    })
                elif 'KEGG' in category:
                    enrichment['kegg_pathways'].append({
                        'pathway': item.get('term', ''),
                        'pvalue': item.get('p_value', 1.0),
                        'fdr': item.get('fdr', 1.0)
                    })

            return enrichment

        except requests.exceptions.RequestException as e:
            logger.error(f"STRING enrichment API error: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    string_db = STRINGDatabase(species=9606, required_score=400)

    # Test single interaction
    result = string_db.get_interaction("EGFR", "GRB2")
    if result:
        print(f"EGFR-GRB2 interaction:")
        print(f"  Combined score: {result['combined_score']}")
        print(f"  Experimental score: {result.get('experimental_score', 0)}")
        print(f"  Exists in STRING: {result['exists_in_string']}")

    # Test enrichment with AutoCoEv predictions
    test_predictions = [
        {'protein_a': 'EGFR', 'protein_b': 'GRB2', 'score': 0.85},
        {'protein_a': 'TP53', 'protein_b': 'MDM2', 'score': 0.92},
        {'protein_a': 'BRCA1', 'protein_b': 'BRCA2', 'score': 0.78}
    ]

    enriched = string_db.enrich_with_string(test_predictions)
    for pred in enriched:
        print(f"\n{pred['protein_a']}-{pred['protein_b']}:")
        print(f"  AutoCoEv score: {pred['score']:.3f}")
        print(f"  STRING score: {pred.get('string_score', 0):.3f}")
        print(f"  Novel: {pred['is_novel']}")
        print(f"  Combined confidence: {pred['combined_confidence']:.3f}")
