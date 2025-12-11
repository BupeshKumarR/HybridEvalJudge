"""
Export service for generating JSON and CSV exports of evaluation and chat data.

This service provides comprehensive export functionality for:
- Evaluation sessions with all judge verdicts and metrics
- Chat sessions with conversation history and evaluations

Requirements: 11.1, 11.2, 11.3
"""
import json
import csv
import io
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from sqlalchemy.orm import Session, selectinload

from ..models import (
    EvaluationSession,
    ChatSession,
    ChatMessage,
    JudgeResult,
    FlaggedIssue,
    VerifierVerdict,
    ClaimVerdict,
    SessionMetadata
)


class ExportService:
    """Service for exporting evaluation and chat data in various formats."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def export_evaluation_json(self, session: EvaluationSession) -> str:
        """
        Export evaluation session as JSON with all data.
        
        Args:
            session: EvaluationSession model instance
            
        Returns:
            JSON string with complete evaluation data
        """
        data = self._build_evaluation_export_data(session)
        return json.dumps(data, indent=2, default=str)
    
    def export_evaluation_csv(self, session: EvaluationSession) -> str:
        """
        Export evaluation session as CSV for tabular data.
        
        Args:
            session: EvaluationSession model instance
            
        Returns:
            CSV string with evaluation data
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Session summary section
        writer.writerow(["=== Evaluation Session Summary ==="])
        writer.writerow([
            "Session ID", "Status", "Consensus Score", "Hallucination Score",
            "Confidence Lower", "Confidence Upper", "Inter-Judge Agreement",
            "Created At", "Completed At"
        ])
        writer.writerow([
            str(session.id),
            session.status,
            session.consensus_score,
            session.hallucination_score,
            session.confidence_interval_lower,
            session.confidence_interval_upper,
            session.inter_judge_agreement,
            session.created_at.isoformat() if session.created_at else "",
            session.completed_at.isoformat() if session.completed_at else ""
        ])
        
        # Source and candidate text
        writer.writerow([])
        writer.writerow(["=== Source Text ==="])
        writer.writerow([session.source_text])
        writer.writerow([])
        writer.writerow(["=== Candidate Output ==="])
        writer.writerow([session.candidate_output])
        
        # Judge results section
        writer.writerow([])
        writer.writerow(["=== Judge Results ==="])
        writer.writerow(["Judge Name", "Score", "Confidence", "Response Time (ms)", "Reasoning"])
        for judge in session.judge_results:
            writer.writerow([
                judge.judge_name,
                judge.score,
                judge.confidence,
                judge.response_time_ms,
                judge.reasoning
            ])
        
        # Flagged issues section
        writer.writerow([])
        writer.writerow(["=== Flagged Issues ==="])
        writer.writerow(["Judge", "Issue Type", "Severity", "Description"])
        for judge in session.judge_results:
            for issue in judge.flagged_issues:
                writer.writerow([
                    judge.judge_name,
                    issue.issue_type,
                    issue.severity,
                    issue.description
                ])
        
        # Verifier verdicts section
        if session.verifier_verdicts:
            writer.writerow([])
            writer.writerow(["=== Verifier Verdicts ==="])
            writer.writerow(["Claim", "Label", "Confidence", "Reasoning"])
            for verdict in session.verifier_verdicts:
                writer.writerow([
                    verdict.claim_text,
                    verdict.label,
                    verdict.confidence,
                    verdict.reasoning
                ])
        
        # Claim verdicts section
        if session.claim_verdicts:
            writer.writerow([])
            writer.writerow(["=== Claim Verdicts ==="])
            writer.writerow(["Claim", "Type", "Verdict", "Confidence", "Judge", "Span Start", "Span End"])
            for claim in session.claim_verdicts:
                writer.writerow([
                    claim.claim_text,
                    claim.claim_type,
                    claim.verdict,
                    claim.confidence,
                    claim.judge_name,
                    claim.text_span_start,
                    claim.text_span_end
                ])
        
        output.seek(0)
        return output.getvalue()
    
    def export_chat_session_json(self, chat_session: ChatSession) -> str:
        """
        Export chat session as JSON with all messages and evaluations.
        
        Args:
            chat_session: ChatSession model instance
            
        Returns:
            JSON string with complete chat session data
        """
        data = self._build_chat_export_data(chat_session)
        return json.dumps(data, indent=2, default=str)
    
    def export_chat_session_csv(self, chat_session: ChatSession) -> str:
        """
        Export chat session as CSV for tabular data.
        
        Args:
            chat_session: ChatSession model instance
            
        Returns:
            CSV string with chat session data
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Session info
        writer.writerow(["=== Chat Session Info ==="])
        writer.writerow(["Session ID", "Ollama Model", "Created At", "Updated At"])
        writer.writerow([
            str(chat_session.id),
            chat_session.ollama_model,
            chat_session.created_at.isoformat() if chat_session.created_at else "",
            chat_session.updated_at.isoformat() if chat_session.updated_at else ""
        ])
        
        # Messages
        writer.writerow([])
        writer.writerow(["=== Messages ==="])
        writer.writerow(["Message ID", "Role", "Content", "Evaluation ID", "Created At"])
        for message in chat_session.messages:
            writer.writerow([
                str(message.id),
                message.role,
                message.content,
                str(message.evaluation_id) if message.evaluation_id else "",
                message.created_at.isoformat() if message.created_at else ""
            ])
        
        # Evaluations for assistant messages
        writer.writerow([])
        writer.writerow(["=== Message Evaluations ==="])
        for message in chat_session.messages:
            if message.evaluation_id and message.evaluation:
                eval_session = message.evaluation
                writer.writerow([f"--- Evaluation for Message {message.id} ---"])
                writer.writerow(["Consensus Score", "Hallucination Score", "Status"])
                writer.writerow([
                    eval_session.consensus_score,
                    eval_session.hallucination_score,
                    eval_session.status
                ])
                
                # Judge results for this evaluation
                if eval_session.judge_results:
                    writer.writerow(["Judge Name", "Score", "Confidence"])
                    for judge in eval_session.judge_results:
                        writer.writerow([
                            judge.judge_name,
                            judge.score,
                            judge.confidence
                        ])
                writer.writerow([])
        
        output.seek(0)
        return output.getvalue()
    
    def _build_evaluation_export_data(self, session: EvaluationSession) -> Dict[str, Any]:
        """Build complete export data structure for evaluation session."""
        return {
            "export_metadata": {
                "export_type": "evaluation_session",
                "exported_at": datetime.utcnow().isoformat(),
                "format_version": "1.0"
            },
            "session": {
                "id": str(session.id),
                "status": session.status,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None
            },
            "content": {
                "source_text": session.source_text,
                "candidate_output": session.candidate_output
            },
            "metrics": {
                "consensus_score": session.consensus_score,
                "hallucination_score": session.hallucination_score,
                "confidence_interval": {
                    "lower": session.confidence_interval_lower,
                    "upper": session.confidence_interval_upper
                },
                "inter_judge_agreement": session.inter_judge_agreement
            },
            "judge_results": [
                {
                    "judge_name": judge.judge_name,
                    "score": judge.score,
                    "confidence": judge.confidence,
                    "reasoning": judge.reasoning,
                    "response_time_ms": judge.response_time_ms,
                    "flagged_issues": [
                        {
                            "issue_type": issue.issue_type,
                            "severity": issue.severity,
                            "description": issue.description,
                            "text_span": {
                                "start": issue.text_span_start,
                                "end": issue.text_span_end
                            } if issue.text_span_start is not None else None
                        }
                        for issue in judge.flagged_issues
                    ]
                }
                for judge in session.judge_results
            ],
            "verifier_verdicts": [
                {
                    "claim_text": verdict.claim_text,
                    "label": verdict.label,
                    "confidence": verdict.confidence,
                    "reasoning": verdict.reasoning
                }
                for verdict in session.verifier_verdicts
            ],
            "claim_verdicts": [
                {
                    "claim_text": claim.claim_text,
                    "claim_type": claim.claim_type,
                    "verdict": claim.verdict,
                    "confidence": claim.confidence,
                    "judge_name": claim.judge_name,
                    "text_span": {
                        "start": claim.text_span_start,
                        "end": claim.text_span_end
                    }
                }
                for claim in session.claim_verdicts
            ],
            "metadata": self._build_session_metadata(session.session_metadata) if session.session_metadata else None
        }
    
    def _build_chat_export_data(self, chat_session: ChatSession) -> Dict[str, Any]:
        """Build complete export data structure for chat session."""
        messages_data = []
        for message in chat_session.messages:
            message_data = {
                "id": str(message.id),
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat() if message.created_at else None,
                "evaluation": None
            }
            
            # Include evaluation data if present
            if message.evaluation_id and message.evaluation:
                message_data["evaluation"] = self._build_evaluation_export_data(message.evaluation)
            
            messages_data.append(message_data)
        
        return {
            "export_metadata": {
                "export_type": "chat_session",
                "exported_at": datetime.utcnow().isoformat(),
                "format_version": "1.0"
            },
            "session": {
                "id": str(chat_session.id),
                "ollama_model": chat_session.ollama_model,
                "created_at": chat_session.created_at.isoformat() if chat_session.created_at else None,
                "updated_at": chat_session.updated_at.isoformat() if chat_session.updated_at else None,
                "message_count": len(chat_session.messages)
            },
            "messages": messages_data
        }
    
    def _build_session_metadata(self, metadata: SessionMetadata) -> Dict[str, Any]:
        """Build metadata export structure."""
        return {
            "total_judges": metadata.total_judges,
            "judges_used": metadata.judges_used,
            "aggregation_strategy": metadata.aggregation_strategy,
            "retrieval_enabled": metadata.retrieval_enabled,
            "num_retrieved_passages": metadata.num_retrieved_passages,
            "num_verifier_verdicts": metadata.num_verifier_verdicts,
            "processing_time_ms": metadata.processing_time_ms,
            "variance": metadata.variance,
            "standard_deviation": metadata.standard_deviation
        }
