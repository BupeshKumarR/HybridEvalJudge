-- LLM Judge Auditor Web Application Database Schema
-- PostgreSQL 15+

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    CONSTRAINT users_username_check CHECK (char_length(username) >= 3),
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Create indexes for users table
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at DESC);

-- Evaluation sessions table
CREATE TABLE evaluation_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_text TEXT NOT NULL,
    candidate_output TEXT NOT NULL,
    consensus_score FLOAT,
    hallucination_score FLOAT,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    inter_judge_agreement FLOAT,
    status VARCHAR(50) DEFAULT 'pending' NOT NULL,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    CONSTRAINT evaluation_sessions_status_check CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')),
    CONSTRAINT evaluation_sessions_consensus_score_check CHECK (consensus_score IS NULL OR (consensus_score >= 0 AND consensus_score <= 100)),
    CONSTRAINT evaluation_sessions_hallucination_score_check CHECK (hallucination_score IS NULL OR (hallucination_score >= 0 AND hallucination_score <= 100)),
    CONSTRAINT evaluation_sessions_confidence_check CHECK (
        (confidence_interval_lower IS NULL AND confidence_interval_upper IS NULL) OR
        (confidence_interval_lower IS NOT NULL AND confidence_interval_upper IS NOT NULL AND confidence_interval_lower <= confidence_interval_upper)
    )
);

-- Create indexes for evaluation_sessions table
CREATE INDEX idx_evaluation_sessions_user_created ON evaluation_sessions(user_id, created_at DESC);
CREATE INDEX idx_evaluation_sessions_consensus_score ON evaluation_sessions(consensus_score);
CREATE INDEX idx_evaluation_sessions_hallucination_score ON evaluation_sessions(hallucination_score);
CREATE INDEX idx_evaluation_sessions_status ON evaluation_sessions(status);
CREATE INDEX idx_evaluation_sessions_created_at ON evaluation_sessions(created_at DESC);
CREATE INDEX idx_evaluation_sessions_user_status ON evaluation_sessions(user_id, status);

-- Full-text search index for source_text and candidate_output
CREATE INDEX idx_evaluation_sessions_source_text_fts ON evaluation_sessions USING gin(to_tsvector('english', source_text));
CREATE INDEX idx_evaluation_sessions_candidate_output_fts ON evaluation_sessions USING gin(to_tsvector('english', candidate_output));

-- Judge results table
CREATE TABLE judge_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES evaluation_sessions(id) ON DELETE CASCADE,
    judge_name VARCHAR(255) NOT NULL,
    score FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    reasoning TEXT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT judge_results_score_check CHECK (score >= 0 AND score <= 100),
    CONSTRAINT judge_results_confidence_check CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT judge_results_response_time_check CHECK (response_time_ms IS NULL OR response_time_ms >= 0)
);

-- Create indexes for judge_results table
CREATE INDEX idx_judge_results_session ON judge_results(session_id);
CREATE INDEX idx_judge_results_judge_name ON judge_results(judge_name);
CREATE INDEX idx_judge_results_score ON judge_results(score);
CREATE INDEX idx_judge_results_session_judge ON judge_results(session_id, judge_name);

-- Flagged issues table
CREATE TABLE flagged_issues (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    judge_result_id UUID NOT NULL REFERENCES judge_results(id) ON DELETE CASCADE,
    issue_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    evidence JSONB,
    text_span_start INTEGER,
    text_span_end INTEGER,
    CONSTRAINT flagged_issues_issue_type_check CHECK (issue_type IN (
        'factual_error', 'hallucination', 'unsupported_claim', 
        'temporal_inconsistency', 'numerical_error', 'bias'
    )),
    CONSTRAINT flagged_issues_severity_check CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT flagged_issues_text_span_check CHECK (
        (text_span_start IS NULL AND text_span_end IS NULL) OR
        (text_span_start IS NOT NULL AND text_span_end IS NOT NULL AND text_span_start <= text_span_end)
    )
);

-- Create indexes for flagged_issues table
CREATE INDEX idx_flagged_issues_judge_result ON flagged_issues(judge_result_id);
CREATE INDEX idx_flagged_issues_issue_type ON flagged_issues(issue_type);
CREATE INDEX idx_flagged_issues_severity ON flagged_issues(severity);
CREATE INDEX idx_flagged_issues_type_severity ON flagged_issues(issue_type, severity);

-- Verifier verdicts table
CREATE TABLE verifier_verdicts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES evaluation_sessions(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    label VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    evidence JSONB,
    reasoning TEXT,
    CONSTRAINT verifier_verdicts_label_check CHECK (label IN ('SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO')),
    CONSTRAINT verifier_verdicts_confidence_check CHECK (confidence >= 0 AND confidence <= 1)
);

-- Create indexes for verifier_verdicts table
CREATE INDEX idx_verifier_verdicts_session ON verifier_verdicts(session_id);
CREATE INDEX idx_verifier_verdicts_label ON verifier_verdicts(label);
CREATE INDEX idx_verifier_verdicts_session_label ON verifier_verdicts(session_id, label);

-- Session metadata table (for advanced analytics)
CREATE TABLE session_metadata (
    session_id UUID PRIMARY KEY REFERENCES evaluation_sessions(id) ON DELETE CASCADE,
    total_judges INTEGER NOT NULL,
    judges_used JSONB NOT NULL,
    aggregation_strategy VARCHAR(100),
    retrieval_enabled BOOLEAN DEFAULT FALSE,
    num_retrieved_passages INTEGER,
    num_verifier_verdicts INTEGER,
    processing_time_ms INTEGER,
    variance FLOAT,
    standard_deviation FLOAT,
    CONSTRAINT session_metadata_total_judges_check CHECK (total_judges > 0),
    CONSTRAINT session_metadata_num_passages_check CHECK (num_retrieved_passages IS NULL OR num_retrieved_passages >= 0),
    CONSTRAINT session_metadata_num_verdicts_check CHECK (num_verifier_verdicts IS NULL OR num_verifier_verdicts >= 0),
    CONSTRAINT session_metadata_processing_time_check CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0)
);

-- Create indexes for session_metadata table
CREATE INDEX idx_session_metadata_aggregation_strategy ON session_metadata(aggregation_strategy);
CREATE INDEX idx_session_metadata_retrieval_enabled ON session_metadata(retrieval_enabled);

-- User preferences table (for configuration management)
CREATE TABLE user_preferences (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    default_judge_models JSONB,
    default_retrieval_enabled BOOLEAN DEFAULT TRUE,
    default_aggregation_strategy VARCHAR(100) DEFAULT 'weighted_average',
    theme VARCHAR(50) DEFAULT 'light',
    notifications_enabled BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for user_preferences
CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create view for session summaries (optimized for history queries)
CREATE VIEW session_summaries AS
SELECT 
    es.id,
    es.user_id,
    es.consensus_score,
    es.hallucination_score,
    es.status,
    es.created_at,
    es.completed_at,
    LEFT(es.source_text, 200) as source_preview,
    LEFT(es.candidate_output, 200) as candidate_preview,
    sm.total_judges,
    sm.aggregation_strategy,
    COUNT(DISTINCT jr.id) as num_judge_results,
    COUNT(DISTINCT vv.id) as num_verifier_verdicts,
    COUNT(DISTINCT fi.id) as num_flagged_issues
FROM evaluation_sessions es
LEFT JOIN session_metadata sm ON es.id = sm.session_id
LEFT JOIN judge_results jr ON es.id = jr.session_id
LEFT JOIN verifier_verdicts vv ON es.id = vv.session_id
LEFT JOIN flagged_issues fi ON jr.id = fi.judge_result_id
GROUP BY es.id, sm.total_judges, sm.aggregation_strategy;

-- Create materialized view for user statistics (for dashboard)
CREATE MATERIALIZED VIEW user_statistics AS
SELECT 
    u.id as user_id,
    u.username,
    COUNT(DISTINCT es.id) as total_evaluations,
    AVG(es.consensus_score) as avg_consensus_score,
    AVG(es.hallucination_score) as avg_hallucination_score,
    MIN(es.created_at) as first_evaluation_at,
    MAX(es.created_at) as last_evaluation_at,
    COUNT(DISTINCT CASE WHEN es.status = 'completed' THEN es.id END) as completed_evaluations,
    COUNT(DISTINCT CASE WHEN es.status = 'failed' THEN es.id END) as failed_evaluations
FROM users u
LEFT JOIN evaluation_sessions es ON u.id = es.user_id
GROUP BY u.id, u.username;

-- Create index on materialized view
CREATE INDEX idx_user_statistics_user_id ON user_statistics(user_id);

-- Create function to refresh user statistics
CREATE OR REPLACE FUNCTION refresh_user_statistics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_statistics;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE users IS 'Stores user account information';
COMMENT ON TABLE evaluation_sessions IS 'Stores evaluation session data with source text, candidate output, and results';
COMMENT ON TABLE judge_results IS 'Stores individual judge verdicts for each evaluation session';
COMMENT ON TABLE flagged_issues IS 'Stores issues flagged by judges during evaluation';
COMMENT ON TABLE verifier_verdicts IS 'Stores claim verification results from the verifier component';
COMMENT ON TABLE session_metadata IS 'Stores metadata and statistics for evaluation sessions';
COMMENT ON TABLE user_preferences IS 'Stores user-specific configuration preferences';

COMMENT ON COLUMN evaluation_sessions.config IS 'JSON configuration used for the evaluation (judge models, retrieval settings, etc.)';
COMMENT ON COLUMN evaluation_sessions.status IS 'Current status of the evaluation: pending, in_progress, completed, failed, cancelled';
COMMENT ON COLUMN judge_results.confidence IS 'Judge confidence level (0-1)';
COMMENT ON COLUMN flagged_issues.evidence IS 'JSON array of evidence supporting the flagged issue';
COMMENT ON COLUMN verifier_verdicts.evidence IS 'JSON array of evidence passages used for verification';
