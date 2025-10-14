"""
LLM-Powered Literature Search and Biological Interpretation
Uses GPT-4 or Claude to validate predictions and generate biological insights
"""

import openai
import anthropic
import logging
from typing import Dict, List, Optional
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMLiteratureSearch:
    """
    LLM-powered literature search and biological interpretation

    Capabilities:
    - Literature validation from PubMed/scientific databases
    - Biological context generation
    - Experimental validation suggestions
    - Novelty assessment
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4",
                 temperature: float = 0.3, max_tokens: int = 1500):
        """
        Initialize LLM client

        Args:
            provider: 'openai' or 'anthropic'
            model: Model name (gpt-4, gpt-4-turbo, claude-3-opus, etc.)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize API clients
        if provider == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {provider}")

        logger.info(f"LLM client initialized: {provider}/{model}")

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API with error handling and retries

        Args:
            prompt: Prompt text

        Returns:
            LLM response text
        """
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert computational biologist and literature researcher specializing in protein-protein interactions."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    return response.choices[0].message.content

                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response.content[0].text

            except Exception as e:
                logger.warning(f"LLM API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"LLM API failed after {max_retries} attempts")
                    return f"Error: Unable to get LLM response - {str(e)}"

        return "Error: LLM API unavailable"

    def validate_interaction(self, protein_a: str, protein_b: str,
                            autocoev_score: float, string_score: float = 0.0) -> Dict:
        """
        Validate protein-protein interaction using literature search

        Args:
            protein_a: First protein name
            protein_b: Second protein name
            autocoev_score: Score from AutoCoEv analysis
            string_score: Score from STRING database (if available)

        Returns:
            Dictionary with literature validation results
        """
        prompt = f"""Analyze the scientific literature for a potential protein-protein interaction between {protein_a} and {protein_b}.

Computational Analysis Results:
- AutoCoEv coevolution score: {autocoev_score:.3f}
- STRING database score: {string_score:.3f}

Please provide:

1. **Literature Support** (HIGH/MEDIUM/LOW/NONE):
   - Search for evidence of direct or indirect interaction
   - Cite specific studies if known (PMID preferred)

2. **Biological Context**:
   - Cellular location of both proteins
   - Known functions of each protein
   - Shared biological pathways or processes

3. **Interaction Mechanism** (if known):
   - Direct binding, complex formation, or regulatory relationship?
   - Known binding domains or interaction interfaces?

4. **Confidence Assessment** (0-100):
   - Overall confidence this interaction exists
   - Explain reasoning

5. **Novelty Status**:
   - KNOWN: Well-established interaction with experimental validation
   - PREDICTED: Computationally predicted, limited experimental evidence
   - NOVEL: No prior evidence in literature

6. **Experimental Validation Suggestions**:
   - 2-3 specific experiments to validate this interaction
   - Expected outcomes if interaction is real

Format your response as structured text with clear sections."""

        response_text = self._call_llm(prompt)

        # Parse response (simple extraction)
        result = {
            'protein_a': protein_a,
            'protein_b': protein_b,
            'autocoev_score': autocoev_score,
            'string_score': string_score,
            'llm_response': response_text,
            'literature_support': self._extract_support_level(response_text),
            'confidence': self._extract_confidence(response_text),
            'novelty': self._extract_novelty(response_text)
        }

        return result

    def _extract_support_level(self, text: str) -> str:
        """Extract literature support level from LLM response"""
        text_lower = text.lower()
        if 'high' in text_lower and 'support' in text_lower:
            return 'HIGH'
        elif 'medium' in text_lower and 'support' in text_lower:
            return 'MEDIUM'
        elif 'low' in text_lower and 'support' in text_lower:
            return 'LOW'
        elif 'none' in text_lower or 'no evidence' in text_lower:
            return 'NONE'
        return 'UNKNOWN'

    def _extract_confidence(self, text: str) -> int:
        """Extract confidence score from LLM response"""
        import re
        # Look for patterns like "confidence: 85" or "85/100"
        patterns = [
            r'confidence[:\s]+(\d+)',
            r'(\d+)/100',
            r'(\d+)%'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))

        return 50  # Default middle confidence

    def _extract_novelty(self, text: str) -> str:
        """Extract novelty status from LLM response"""
        text_lower = text.lower()
        if 'known' in text_lower or 'well-established' in text_lower:
            return 'KNOWN'
        elif 'predicted' in text_lower or 'computational' in text_lower:
            return 'PREDICTED'
        elif 'novel' in text_lower or 'no prior' in text_lower:
            return 'NOVEL'
        return 'UNKNOWN'

    def generate_biological_interpretation(self, interaction_data: Dict) -> str:
        """
        Generate comprehensive biological interpretation for an interaction

        Args:
            interaction_data: Dictionary with interaction details

        Returns:
            Formatted biological interpretation text
        """
        protein_a = interaction_data['protein_a']
        protein_b = interaction_data['protein_b']
        score = interaction_data.get('score', 0)

        prompt = f"""Generate a comprehensive biological interpretation for the predicted interaction between {protein_a} and {protein_b} (confidence score: {score:.3f}).

Please provide a detailed analysis suitable for a research paper supplementary material:

1. **Biological Significance**: Why is this interaction biologically meaningful?

2. **Functional Context**: What cellular processes or pathways might this interaction affect?

3. **Disease Relevance**: Are these proteins implicated in any diseases? Could this interaction be therapeutically relevant?

4. **Evolutionary Conservation**: Is this interaction likely conserved across species?

5. **Molecular Mechanism**: What might be the molecular basis for this interaction?

Write in clear, academic prose suitable for biologists. Be specific but accessible."""

        interpretation = self._call_llm(prompt)

        return interpretation

    def batch_validate_interactions(self, interactions: List[Dict],
                                   max_workers: int = 5) -> List[Dict]:
        """
        Validate multiple interactions in parallel

        Args:
            interactions: List of interaction dictionaries
            max_workers: Number of parallel LLM requests

        Returns:
            List of validated interactions with LLM insights
        """
        logger.info(f"Validating {len(interactions)} interactions with LLM...")

        validated = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.validate_interaction,
                    interaction['protein_a'],
                    interaction['protein_b'],
                    interaction.get('score', 0),
                    interaction.get('string_score', 0)
                ): interaction
                for interaction in interactions
            }

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    original = futures[future]

                    # Merge LLM results with original data
                    validated.append({
                        **original,
                        **result
                    })

                    if i % 10 == 0:
                        logger.info(f"Validated {i}/{len(interactions)} interactions")

                except Exception as e:
                    logger.error(f"Error validating interaction: {e}")

        logger.info(f"LLM validation complete")

        return validated

    def prioritize_for_validation(self, interactions: List[Dict],
                                  top_n: int = 20) -> List[Dict]:
        """
        Prioritize which interactions to validate with LLM

        Focus on:
        - High AutoCoEv scores
        - Novel predictions (not in STRING)
        - Interesting biological context

        Args:
            interactions: List of all interactions
            top_n: Number of top interactions to validate

        Returns:
            Prioritized subset of interactions
        """
        # Score interactions for prioritization
        scored = []
        for interaction in interactions:
            priority_score = 0

            # High AutoCoEv score
            priority_score += interaction.get('score', 0) * 50

            # Novel (not in STRING)
            if interaction.get('is_novel', False):
                priority_score += 30

            # High STRING score (interesting if both methods agree)
            priority_score += interaction.get('string_score', 0) * 20

            scored.append((priority_score, interaction))

        # Sort by priority score
        scored.sort(reverse=True, key=lambda x: x[0])

        # Return top N
        return [interaction for _, interaction in scored[:top_n]]


if __name__ == "__main__":
    # Example usage
    # Note: Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable

    llm = LLMLiteratureSearch(provider="openai", model="gpt-4")

    # Test validation
    result = llm.validate_interaction(
        protein_a="EGFR",
        protein_b="GRB2",
        autocoev_score=0.87,
        string_score=0.95
    )

    print(f"Interaction: {result['protein_a']}-{result['protein_b']}")
    print(f"Literature Support: {result['literature_support']}")
    print(f"Confidence: {result['confidence']}/100")
    print(f"Novelty: {result['novelty']}")
    print(f"\nFull LLM Response:\n{result['llm_response']}")
