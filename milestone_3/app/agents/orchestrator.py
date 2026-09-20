"""Multi-Agent Orchestrator for Milestone 3 (Full Pipeline).

Coordinates the sequential multi-agent query pipeline:
User Query -> Conversation Memory (Anaphora/Contextualization) -> Query Understanding 
-> (Clarification Loop) -> Retrieval -> Response Generation (with Transparency & TTS synthesis) -> Record Turn.
Supports multi-turn sessions, context continuity, and refinement via `resolve_clarification`.
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

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    """Coordinates multi-agent query understanding, memory contextualization, clarification, retrieval, and grounded response generation."""

    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        query_understanding_agent: Optional[QueryUnderstandingAgent] = None,
        response_generation_agent: Optional[ResponseGenerationAgent] = None,
        clarification_agent: Optional[ClarificationAgent] = None,
        memory_agent: Optional[ConversationMemoryAgent] = None
    ):
        self.clarification_agent = clarification_agent or ClarificationAgent()
        self.query_understanding_agent = query_understanding_agent or QueryUnderstandingAgent(
            clarification_agent=self.clarification_agent
        )
        self.retrieval_agent = retrieval_agent
        self.response_generation_agent = response_generation_agent or ResponseGenerationAgent()
        self.memory_agent = memory_agent or ConversationMemoryAgent()

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ) -> OrchestrationResult:
        """
        Execute multi-agent pipeline with conversation memory context, clarification routing, retrieval, and transparency.
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Conversation Memory & Contextualization
            session: Optional[ConversationSession] = None
            effective_query = query
            if session_id:
                session = self.memory_agent.get_or_create_session(session_id)
                effective_query = self.memory_agent.contextualize_query(query, session)

            # Step 2: Query Understanding & Clarification Detection
            query_analysis = self.query_understanding_agent.analyze(effective_query)

            # Step 3: Route Decision (Clarification Loop)
            if query_analysis.route_to == RoutingTarget.CLARIFICATION:
                clarification_response = self.clarification_agent.handle(query_analysis)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return OrchestrationResult(
                    query=query,
                    response=clarification_response,
                    retrieval_result=None,
                    execution_time_ms=round(elapsed_ms, 2),
                    status="clarification_requested",
                    clarification_request=clarification_response.clarification_request,
                    session_id=session_id
                )

            # Step 4: Retrieval Agent (Vector Search & Relevance Filtering)
            retrieval_result = self.retrieval_agent.retrieve(
                query_input=query_analysis,
                top_k=top_k,
                threshold=confidence_threshold
            )

            # Step 5: Response Generation Agent (Grounded Synthesis, Transparency, & Speech Sanitization)
            agent_response = self.response_generation_agent.generate(
                query_analysis=query_analysis,
                retrieval_result=retrieval_result
            )

            # Step 6: Record Turn in Conversation Memory
            if session:
                turn = ConversationTurn(
                    query=query,
                    effective_query=effective_query,
                    response=agent_response.answer,
                    confidence_score=agent_response.confidence_score,
                    sources=agent_response.sources,
                    key_entities=query_analysis.entities,
                    transparency=agent_response.transparency
                )
                self.memory_agent.record_turn(session, turn)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return OrchestrationResult(
                query=query,
                response=agent_response,
                retrieval_result=retrieval_result,
                execution_time_ms=round(elapsed_ms, 2),
                status="success",
                session_id=session_id
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
                error=str(exc),
                session_id=session_id
            )

    def resolve_clarification(
        self,
        clarification_id: str,
        user_response: str,
        session_id: Optional[str] = None,
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

            # Step 5: Record Turn in Memory if session exists
            if session_id:
                session = self.memory_agent.get_or_create_session(session_id)
                turn = ConversationTurn(
                    query=clarification_req.original_query if clarification_req else refined_query,
                    effective_query=refined_query,
                    response=agent_response.answer,
                    confidence_score=agent_response.confidence_score,
                    sources=agent_response.sources,
                    key_entities=query_analysis.entities,
                    transparency=agent_response.transparency
                )
                self.memory_agent.record_turn(session, turn)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            orig_q = clarification_req.original_query if clarification_req else refined_query

            return OrchestrationResult(
                query=orig_q,
                response=agent_response,
                retrieval_result=retrieval_result,
                execution_time_ms=round(elapsed_ms, 2),
                status="clarified_success",
                clarification_request=clarification_req,
                session_id=session_id
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
                error=str(exc),
                session_id=session_id
            )
