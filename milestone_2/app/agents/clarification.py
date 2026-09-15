"""Clarification Agent for Milestone 2.

Generates targeted clarification requests when queries are ambiguous,
underspecified, or missing domain entities.
"""
from typing import Optional
from .models import QueryAnalysisResult, AgentResponse, ConfidenceLevel

class ClarificationAgent:
    """Handles ambiguous queries by prompting the user for missing details."""

    def handle(self, query_analysis: QueryAnalysisResult) -> AgentResponse:
        """Generate clarification message and return structured AgentResponse."""
        query = query_analysis.query.strip()
        
        if not query:
            msg = "Please enter a specific question or topic (e.g., 'What is overfitting in Machine Learning?' or 'Compare TCP vs UDP')."
        elif len(query.split()) <= 2 and not query_analysis.entities:
            msg = f"Your query '{query}' is too brief. Could you specify which concept, algorithm, or protocol you would like to explore?"
        else:
            msg = (
                f"Your question appears ambiguous ({query_analysis.reasoning.lower()}). "
                "Please provide more specific details or name the exact technology/protocol you are asking about."
            )

        return AgentResponse(
            answer=msg,
            confidence_score=query_analysis.confidence,
            confidence_level=ConfidenceLevel.NONE,
            sources=[],
            query_analysis=query_analysis,
            has_sufficient_evidence=False,
            requires_clarification=True
        )
