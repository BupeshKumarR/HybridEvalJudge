"""
PDF export service for evaluation results.
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from io import BytesIO
from datetime import datetime
from typing import Optional

from ..models import EvaluationSession


class PDFExportService:
    """Service for generating PDF reports of evaluation results."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4F46E5'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1F2937'),
            spaceAfter=12,
            spaceBefore=20,
            borderWidth=0,
            borderColor=colors.HexColor('#E5E7EB'),
            borderPadding=5
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#374151'),
            spaceAfter=8,
            spaceBefore=12
        ))
        
        # Score style
        self.styles.add(ParagraphStyle(
            name='ScoreText',
            parent=self.styles['Normal'],
            fontSize=36,
            textColor=colors.HexColor('#4F46E5'),
            alignment=TA_CENTER,
            spaceAfter=10
        ))
    
    def generate_pdf(self, session: EvaluationSession) -> BytesIO:
        """
        Generate a PDF report for an evaluation session.
        
        Args:
            session: EvaluationSession model instance
            
        Returns:
            BytesIO buffer containing the PDF
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build the PDF content
        story = []
        
        # Title page
        story.extend(self._build_title_page(session))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._build_executive_summary(session))
        story.append(Spacer(1, 0.3 * inch))
        
        # Judge results
        story.extend(self._build_judge_results(session))
        story.append(Spacer(1, 0.3 * inch))
        
        # Confidence metrics
        story.extend(self._build_confidence_metrics(session))
        story.append(Spacer(1, 0.3 * inch))
        
        # Hallucination analysis
        story.extend(self._build_hallucination_analysis(session))
        story.append(Spacer(1, 0.3 * inch))
        
        # Verifier verdicts
        if session.verifier_verdicts:
            story.extend(self._build_verifier_verdicts(session))
            story.append(Spacer(1, 0.3 * inch))
        
        # Statistical metrics
        story.extend(self._build_statistical_metrics(session))
        story.append(PageBreak())
        
        # Source text and candidate output
        story.extend(self._build_text_content(session))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    
    def _build_title_page(self, session: EvaluationSession) -> list:
        """Build the title page."""
        elements = []
        
        # Title
        elements.append(Spacer(1, 2 * inch))
        elements.append(Paragraph(
            "LLM Judge Auditor",
            self.styles['CustomTitle']
        ))
        elements.append(Paragraph(
            "Evaluation Report",
            self.styles['CustomTitle']
        ))
        
        elements.append(Spacer(1, 0.5 * inch))
        
        # Session info
        # Handle status - it might be a string or enum
        status_text = session.status.value.upper() if hasattr(session.status, 'value') else str(session.status).upper()
        
        session_info = f"""
        <para alignment="center">
        <b>Session ID:</b> {session.id}<br/>
        <b>Date:</b> {session.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>
        <b>Status:</b> {status_text}
        </para>
        """
        elements.append(Paragraph(session_info, self.styles['Normal']))
        
        elements.append(Spacer(1, 1 * inch))
        
        # Consensus score (large)
        if session.consensus_score is not None:
            elements.append(Paragraph(
                f"{session.consensus_score:.1f}",
                self.styles['ScoreText']
            ))
            elements.append(Paragraph(
                "Consensus Score",
                self.styles['Normal']
            ))
        
        return elements
    
    def _build_executive_summary(self, session: EvaluationSession) -> list:
        """Build executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Create summary table
        data = [
            ['Metric', 'Value'],
            ['Consensus Score', f"{session.consensus_score:.2f}" if session.consensus_score else "N/A"],
            ['Hallucination Score', f"{session.hallucination_score:.2f}" if session.hallucination_score else "N/A"],
            ['Confidence Interval', 
             f"[{session.confidence_interval_lower:.2f}, {session.confidence_interval_upper:.2f}]" 
             if session.confidence_interval_lower and session.confidence_interval_upper else "N/A"],
            ['Inter-Judge Agreement', f"{session.inter_judge_agreement:.3f}" if session.inter_judge_agreement else "N/A"],
            ['Number of Judges', str(len(session.judge_results))],
            ['Number of Verifier Verdicts', str(len(session.verifier_verdicts))],
        ]
        
        if session.session_metadata:
            data.append(['Processing Time', f"{session.session_metadata.processing_time_ms / 1000:.2f}s" 
                        if session.session_metadata.processing_time_ms else "N/A"])
        
        table = Table(data, colWidths=[3 * inch, 3 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))
        
        elements.append(table)
        
        return elements
    
    def _build_judge_results(self, session: EvaluationSession) -> list:
        """Build judge results section."""
        elements = []
        
        elements.append(Paragraph("Judge Results", self.styles['SectionHeader']))
        
        for judge_result in session.judge_results:
            elements.append(Paragraph(
                f"<b>{judge_result.judge_name}</b>",
                self.styles['SubsectionHeader']
            ))
            
            # Judge metrics table
            data = [
                ['Score', f"{judge_result.score:.2f}"],
                ['Confidence', f"{judge_result.confidence * 100:.1f}%"],
                ['Response Time', f"{judge_result.response_time_ms}ms" if judge_result.response_time_ms else "N/A"],
            ]
            
            table = Table(data, colWidths=[2 * inch, 4 * inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.1 * inch))
            
            # Reasoning
            if judge_result.reasoning:
                elements.append(Paragraph("<b>Reasoning:</b>", self.styles['Normal']))
                elements.append(Paragraph(
                    judge_result.reasoning,
                    self.styles['Normal']
                ))
                elements.append(Spacer(1, 0.1 * inch))
            
            # Flagged issues
            if judge_result.flagged_issues:
                elements.append(Paragraph(
                    f"<b>Flagged Issues ({len(judge_result.flagged_issues)}):</b>",
                    self.styles['Normal']
                ))
                
                for issue in judge_result.flagged_issues:
                    issue_text = f"• <b>{issue.issue_type}</b> ({issue.severity}): {issue.description}"
                    elements.append(Paragraph(issue_text, self.styles['Normal']))
                
                elements.append(Spacer(1, 0.1 * inch))
            
            elements.append(Spacer(1, 0.2 * inch))
        
        return elements
    
    def _build_confidence_metrics(self, session: EvaluationSession) -> list:
        """Build confidence metrics section."""
        elements = []
        
        elements.append(Paragraph("Confidence Metrics", self.styles['SectionHeader']))
        
        if session.confidence_interval_lower and session.confidence_interval_upper:
            data = [
                ['Metric', 'Value'],
                ['Confidence Interval Lower', f"{session.confidence_interval_lower:.2f}"],
                ['Confidence Interval Upper', f"{session.confidence_interval_upper:.2f}"],
                ['Interval Width', f"{session.confidence_interval_upper - session.confidence_interval_lower:.2f}"],
            ]
            
            if session.inter_judge_agreement is not None:
                data.append(['Inter-Judge Agreement', f"{session.inter_judge_agreement:.3f}"])
            
            table = Table(data, colWidths=[3 * inch, 3 * inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5E7EB')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ]))
            
            elements.append(table)
        else:
            elements.append(Paragraph("No confidence metrics available.", self.styles['Normal']))
        
        return elements
    
    def _build_hallucination_analysis(self, session: EvaluationSession) -> list:
        """Build hallucination analysis section."""
        elements = []
        
        elements.append(Paragraph("Hallucination Analysis", self.styles['SectionHeader']))
        
        if session.hallucination_score is not None:
            elements.append(Paragraph(
                f"<b>Overall Hallucination Score:</b> {session.hallucination_score:.2f}",
                self.styles['Normal']
            ))
            elements.append(Spacer(1, 0.1 * inch))
            
            # Count flagged issues by type
            issue_counts = {}
            for judge_result in session.judge_results:
                for issue in judge_result.flagged_issues:
                    issue_type = issue.issue_type
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            if issue_counts:
                elements.append(Paragraph("<b>Issues by Type:</b>", self.styles['Normal']))
                
                data = [['Issue Type', 'Count']]
                for issue_type, count in sorted(issue_counts.items()):
                    data.append([issue_type.replace('_', ' ').title(), str(count)])
                
                table = Table(data, colWidths=[3 * inch, 3 * inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5E7EB')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                
                elements.append(table)
        else:
            elements.append(Paragraph("No hallucination analysis available.", self.styles['Normal']))
        
        return elements
    
    def _build_verifier_verdicts(self, session: EvaluationSession) -> list:
        """Build verifier verdicts section."""
        elements = []
        
        elements.append(Paragraph("Verifier Verdicts", self.styles['SectionHeader']))
        
        for verdict in session.verifier_verdicts:
            elements.append(Paragraph(
                f"<b>Claim:</b> {verdict.claim_text}",
                self.styles['Normal']
            ))
            elements.append(Paragraph(
                f"<b>Label:</b> {verdict.label} (Confidence: {verdict.confidence * 100:.1f}%)",
                self.styles['Normal']
            ))
            
            if verdict.reasoning:
                elements.append(Paragraph(
                    f"<b>Reasoning:</b> {verdict.reasoning}",
                    self.styles['Normal']
                ))
            
            elements.append(Spacer(1, 0.15 * inch))
        
        return elements
    
    def _build_statistical_metrics(self, session: EvaluationSession) -> list:
        """Build statistical metrics section."""
        elements = []
        
        elements.append(Paragraph("Statistical Metrics", self.styles['SectionHeader']))
        
        if session.session_metadata:
            data = [
                ['Metric', 'Value'],
            ]
            
            if session.session_metadata.variance is not None:
                data.append(['Variance', f"{session.session_metadata.variance:.4f}"])
            
            if session.session_metadata.standard_deviation is not None:
                data.append(['Standard Deviation', f"{session.session_metadata.standard_deviation:.4f}"])
            
            if session.session_metadata.processing_time_ms is not None:
                data.append(['Processing Time', f"{session.session_metadata.processing_time_ms / 1000:.2f}s"])
            
            if len(data) > 1:
                table = Table(data, colWidths=[3 * inch, 3 * inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5E7EB')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ]))
                
                elements.append(table)
            else:
                elements.append(Paragraph("No statistical metrics available.", self.styles['Normal']))
        else:
            elements.append(Paragraph("No statistical metrics available.", self.styles['Normal']))
        
        return elements
    
    def _build_text_content(self, session: EvaluationSession) -> list:
        """Build source text and candidate output section."""
        elements = []
        
        elements.append(Paragraph("Source Text", self.styles['SectionHeader']))
        elements.append(Paragraph(session.source_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))
        
        elements.append(Paragraph("Candidate Output", self.styles['SectionHeader']))
        elements.append(Paragraph(session.candidate_output, self.styles['Normal']))
        
        return elements
