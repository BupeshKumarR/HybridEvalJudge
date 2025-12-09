"""
Simple script to verify database models are correctly defined.
"""
import sys
sys.path.insert(0, '.')

from app.database import Base
from app.models import (
    User, EvaluationSession, JudgeResult, FlaggedIssue,
    VerifierVerdict, SessionMetadata, UserPreference
)

def verify_models():
    """Verify all models are correctly defined."""
    print("Verifying database models...")
    
    # Check that all models are registered with Base
    tables = Base.metadata.tables
    expected_tables = [
        'users',
        'evaluation_sessions',
        'judge_results',
        'flagged_issues',
        'verifier_verdicts',
        'session_metadata',
        'user_preferences'
    ]
    
    print(f"\nExpected tables: {len(expected_tables)}")
    print(f"Found tables: {len(tables)}")
    
    for table_name in expected_tables:
        if table_name in tables:
            print(f"✓ {table_name}")
        else:
            print(f"✗ {table_name} - MISSING")
            return False
    
    # Check model attributes
    print("\nVerifying model attributes...")
    
    # User model
    user_attrs = ['id', 'username', 'email', 'password_hash', 'created_at', 'last_login']
    for attr in user_attrs:
        if hasattr(User, attr):
            print(f"✓ User.{attr}")
        else:
            print(f"✗ User.{attr} - MISSING")
            return False
    
    # EvaluationSession model
    session_attrs = ['id', 'user_id', 'source_text', 'candidate_output', 'consensus_score', 
                     'hallucination_score', 'status', 'created_at']
    for attr in session_attrs:
        if hasattr(EvaluationSession, attr):
            print(f"✓ EvaluationSession.{attr}")
        else:
            print(f"✗ EvaluationSession.{attr} - MISSING")
            return False
    
    # JudgeResult model
    judge_attrs = ['id', 'session_id', 'judge_name', 'score', 'confidence', 'reasoning']
    for attr in judge_attrs:
        if hasattr(JudgeResult, attr):
            print(f"✓ JudgeResult.{attr}")
        else:
            print(f"✗ JudgeResult.{attr} - MISSING")
            return False
    
    # Check relationships
    print("\nVerifying relationships...")
    
    if hasattr(User, 'evaluation_sessions'):
        print("✓ User.evaluation_sessions relationship")
    else:
        print("✗ User.evaluation_sessions relationship - MISSING")
        return False
    
    if hasattr(EvaluationSession, 'user'):
        print("✓ EvaluationSession.user relationship")
    else:
        print("✗ EvaluationSession.user relationship - MISSING")
        return False
    
    if hasattr(EvaluationSession, 'judge_results'):
        print("✓ EvaluationSession.judge_results relationship")
    else:
        print("✗ EvaluationSession.judge_results relationship - MISSING")
        return False
    
    if hasattr(JudgeResult, 'flagged_issues'):
        print("✓ JudgeResult.flagged_issues relationship")
    else:
        print("✗ JudgeResult.flagged_issues relationship - MISSING")
        return False
    
    print("\n✅ All models verified successfully!")
    return True

if __name__ == "__main__":
    try:
        success = verify_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error verifying models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
