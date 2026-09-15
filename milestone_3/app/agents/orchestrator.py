"""Multi-Agent Orchestrator for Milestone 3 (M3.1 Clarification Loop).

Coordinates the sequential query pipeline:
User Query -> Query Understanding -> (Clarification Loop) -> Retrieval -> Response Generation -> Grounded Answer.
Supports re-entry and query refinement via `resolve_clarification`.
"""
import time
import logging
from typing import Optional, Any
from .models import (
    RoutingTarget,
    QueryAnalysisResult,
    RetrievalResult,
    AgentResponse,
    OrchestrationResult,
    ConfidenceLevel,
    ClarificationRequest
)
from .query_understanding import QueryUnderstandingAgent
from .retrieval import RetrievalAgent
from .response_generation import ResponseGenerationAgent
from .clarification import ClarificationAgent

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    """Coordinates multi-agent query understanding, clarification loops, retrieval, and grounded response generation."""

    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        query_understanding_agent: Optional[QueryUnderstandingAgent] = None,
        response_generation_agent: Optional[ResponseGenerationAgent] = None,
        clarification_agent: Optional[ClarificationAgent] = None
    ):
        self.clarification_agent = clarification_agent or ClarificationAgent()
        self.query_understanding_agent = query_understanding_agent or QueryUnderstandingAgent(
            clarification_agent=self.clarification_agent
        )
        self.retrieval_agent = retrieval_agent
        self.response_generation_agent = response_generation_agent or ResponseGenerationAgent()

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ) -> OrchestrationResult:
        """
        Execute initial query understanding and determine whether to trigger clarification or proceed to retrieval.
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Query Understanding & Clarification Detection
            query_analysis = self.query_understanding_agent.analyze(query)

            # Step 2: Route Decision
            if query_analysis.route_to == RoutingTarget.CLARIFICATION:
                clarification_response = self.clarification_agent.handle(query_analysis)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return OrchestrationResult(
                    query=query,
                    response=clarification_response,
                    retrieval_result=None,
                    execution_time_ms=round(elapsed_ms, 2),
                    status="clarification_requested",
                    clarification_request=clarification_response.clarification_request
                )

            # Step 3: Retrieval Agent (Vector Search & Relevance Filtering)
            retrieval_result = self.retrieval_agent.retrieve(
                query_input=query_analysis,
                top_k=top_k,
                threshold=confidence_threshold
            )

            # Step 4: Response Generation Agent (Grounded Synthesis & Attribution)
            agent_response = self.response_generation_agent.generate(
                query_analysis=query_analysis,
                retrieval_result=retrieval_result
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return OrchestrationResult(
                query=query,
                response=agent_response,
                retrieval_result=retrieval_result,
                execution_time_ms=round(elapsed_ms, 2),
                status="success"
            )

        except Exception as exc:
            logger.error(f"Error during multi-agent orchestration: {exc}", exc_info=True)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            fallback_qa = self.query_understanding_agent.analyze(query)
            fallback_response = AgentResponse(
                answer="An unexpected error occurred while processing your query. Please check your system configuration or try again.",
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.NONE,
                sources=[],
                query_analysis=fallback_qa,
                has_sufficient_evidence=False,
                requires_clarification=False
            )
            return OrchestrationResult(
                query=query,
                response=fallback_response,
                retrieval_result=None,
                execution_time_ms=round(elapsed_ms, 2),
                status="error",
                error=str(exc)
            )

    def resolve_clarification(
        self,
        clarification_id: str,
        user_response: str,
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ) -> OrchestrationResult:
        """
        Fuses the original query with user clarification, refines the query, and re-enters the retrieval pipeline.
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Resolve pending clarification and synthesize refined query
            refined_query, clarification_req = self.clarification_agent.resolve_clarification(
                clarification_id=clarification_id,
                user_response=user_response
            )

            # Step 2: Analyze refined query intent
            query_analysis = self.query_understanding_agent.analyze(refined_query)

            # Step 3: Execute Retrieval on refined query
            retrieval_result = self.retrieval_agent.retrieve(
                query_input=query_analysis,
                top_k=top_k,
                threshold=confidence_threshold
            )

            # Step 4: Generate grounded response
            agent_response = self.response_generation_agent.generate(
                query_analysis=query_analysis,
                retrieval_result=retrieval_result,
                is_clarified_resolution=True,
                refined_query=refined_query
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            orig_q = clarification_req.original_query if clarification_req else refined_query

            return OrchestrationResult(
                query=orig_q,
                response=agent_response,
                retrieval_result=retrieval_result,
                execution_time_ms=round(elapsed_ms, 2),
                status="clarified_success",
                clarification_request=clarification_req
            )

        except Exception as exc:
            logger.error(f"Error during clarification resolution orchestration: {exc}", exc_info=True)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            fallback_qa = self.query_understanding_agent.analyze(user_response)
            fallback_response = AgentResponse(
                answer=f"Could not resolve clarification: {exc}",
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.NONE,
                sources=[],
                query_analysis=fallback_qa,
                has_sufficient_evidence=False,
                requires_clarification=False
            )
            return OrchestrationResult(
                query=user_response,
                response=fallback_response,
                retrieval_result=None,
                execution_time_ms=round(elapsed_ms, 2),
                status="error",
                error=str(exc)
            )
