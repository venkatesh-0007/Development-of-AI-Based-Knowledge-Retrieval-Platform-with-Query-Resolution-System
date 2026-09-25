"""Multi-Agent Orchestrator for Milestone 4 (Full Pipeline with Query Analytics Integration).

Coordinates the sequential multi-agent query pipeline:
User Query -> Conversation Memory -> Query Understanding -> (Clarification Loop) -> Retrieval
-> Response Generation (Transparency & Speech) -> Record Memory Turn -> Log to Analytics Telemetry.
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
    ClarificationRequest,
    ConversationTurn,
    ConversationSession
)
from .query_understanding import QueryUnderstandingAgent
from .retrieval import RetrievalAgent
from .response_generation import ResponseGenerationAgent
from .clarification import ClarificationAgent
from .memory import ConversationMemoryAgent
from ..analytics.tracker import AnalyticsTracker

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    """Coordinates multi-agent query understanding, memory, clarification, retrieval, and analytics telemetry."""

    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        query_understanding_agent: Optional[QueryUnderstandingAgent] = None,
        response_generation_agent: Optional[ResponseGenerationAgent] = None,
        clarification_agent: Optional[ClarificationAgent] = None,
        memory_agent: Optional[ConversationMemoryAgent] = None,
        analytics_tracker: Optional[AnalyticsTracker] = None
    ):
        self.clarification_agent = clarification_agent or ClarificationAgent()
        self.query_understanding_agent = query_understanding_agent or QueryUnderstandingAgent(
            clarification_agent=self.clarification_agent
        )
        self.retrieval_agent = retrieval_agent
        self.response_generation_agent = response_generation_agent or ResponseGenerationAgent()
        self.memory_agent = memory_agent or ConversationMemoryAgent()
        self.analytics_tracker = analytics_tracker

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        active_domain: Optional[str] = None,
        input_modality: str = "text"
    ) -> OrchestrationResult:
        start_time = time.perf_counter()

        try:
            # Step 1: Conversation Memory & Contextualization
            session: Optional[ConversationSession] = None
            effective_query = query
            if session_id:
                session = self.memory_agent.get_or_create_session(session_id)
                effective_query = self.memory_agent.contextualize_query(query, session)

            # Step 2: Query Understanding & Clarification Detection
            query_analysis = self.query_understanding_agent.analyze(
                effective_query,
                active_domain=active_domain or (session.active_domain if session else None)
            )

            # Step 3: Route Decision (Clarification Loop)
            if query_analysis.route_to == RoutingTarget.CLARIFICATION:
                clarification_response = self.clarification_agent.handle(query_analysis)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                res = OrchestrationResult(
                    query=query,
                    response=clarification_response,
                    retrieval_result=None,
                    execution_time_ms=round(elapsed_ms, 2),
                    status="clarification_requested",
                    clarification_request=clarification_response.clarification_request,
                    session_id=session_id,
                    input_modality=input_modality
                )
                if self.analytics_tracker:
                    self.analytics_tracker.record_orchestration(res, input_modality=input_modality)
                return res

            # Step 4: Retrieval Agent
            retrieval_result = self.retrieval_agent.retrieve(
                query_input=query_analysis,
                top_k=top_k,
                threshold=confidence_threshold
            )

            # Step 5: Response Generation Agent
            agent_response = self.response_generation_agent.generate(
                query_analysis=query_analysis,
                retrieval_result=retrieval_result
            )

            # Step 6: Record Turn in Memory
            if session:
                turn = ConversationTurn(
                    query=query,
                    effective_query=effective_query,
                    response=agent_response.answer,
                    confidence_score=agent_response.confidence_score,
                    sources=agent_response.sources,
                    key_entities=query_analysis.entities,
                    transparency=agent_response.transparency,
                    domain=query_analysis.domain
                )
                self.memory_agent.record_turn(session, turn)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            res = OrchestrationResult(
                query=query,
                response=agent_response,
                retrieval_result=retrieval_result,
                execution_time_ms=round(elapsed_ms, 2),
                status="success",
                session_id=session_id,
                input_modality=input_modality
            )

            # Step 7: Record Analytics Telemetry
            if self.analytics_tracker:
                self.analytics_tracker.record_orchestration(res, input_modality=input_modality)

            return res

        except Exception as exc:
            logger.error(f"Error during orchestration: {exc}", exc_info=True)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            fallback_qa = self.query_understanding_agent.analyze(query)
            fallback_response = AgentResponse(
                answer="An unexpected error occurred while processing your query. Please verify configuration or retry.",
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.NONE,
                sources=[],
                query_analysis=fallback_qa,
                has_sufficient_evidence=False,
                requires_clarification=False
            )
            res = OrchestrationResult(
                query=query,
                response=fallback_response,
                retrieval_result=None,
                execution_time_ms=round(elapsed_ms, 2),
                status="error",
                error=str(exc),
                session_id=session_id,
                input_modality=input_modality
            )
            if self.analytics_tracker:
                self.analytics_tracker.record_orchestration(res, input_modality=input_modality)
            return res

    def resolve_clarification(
        self,
        clarification_id: str,
        user_response: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        input_modality: str = "text"
    ) -> OrchestrationResult:
        start_time = time.perf_counter()

        try:
            # Step 1: Disambiguate and synthesize refined query
            refined_query, clar_req = self.clarification_agent.resolve_clarification(
                clarification_id=clarification_id,
                user_response=user_response
            )

            # Step 2: Query Understanding on refined query
            query_analysis = self.query_understanding_agent.analyze(refined_query)

            # Step 3: Retrieval
            retrieval_result = self.retrieval_agent.retrieve(
                query_input=query_analysis,
                top_k=top_k,
                threshold=confidence_threshold
            )

            # Step 4: Grounded Synthesis
            agent_response = self.response_generation_agent.generate(
                query_analysis=query_analysis,
                retrieval_result=retrieval_result,
                is_clarified_resolution=True,
                refined_query=refined_query
            )

            # Step 5: Record Turn in Memory
            if session_id:
                session = self.memory_agent.get_or_create_session(session_id)
                turn = ConversationTurn(
                    query=clar_req.original_query if clar_req else refined_query,
                    effective_query=refined_query,
                    response=agent_response.answer,
                    confidence_score=agent_response.confidence_score,
                    sources=agent_response.sources,
                    key_entities=query_analysis.entities,
                    transparency=agent_response.transparency,
                    domain=query_analysis.domain
                )
                self.memory_agent.record_turn(session, turn)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            orig_q = clar_req.original_query if clar_req else refined_query

            res = OrchestrationResult(
                query=orig_q,
                response=agent_response,
                retrieval_result=retrieval_result,
                execution_time_ms=round(elapsed_ms, 2),
                status="clarified_success",
                clarification_request=clar_req,
                session_id=session_id,
                input_modality=input_modality
            )

            if self.analytics_tracker:
                self.analytics_tracker.record_orchestration(res, input_modality=input_modality)

            return res

        except Exception as exc:
            logger.error(f"Error resolving clarification: {exc}", exc_info=True)
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
            res = OrchestrationResult(
                query=user_response,
                response=fallback_response,
                retrieval_result=None,
                execution_time_ms=round(elapsed_ms, 2),
                status="error",
                error=str(exc),
                session_id=session_id,
                input_modality=input_modality
            )
            if self.analytics_tracker:
                self.analytics_tracker.record_orchestration(res, input_modality=input_modality)
            return res
