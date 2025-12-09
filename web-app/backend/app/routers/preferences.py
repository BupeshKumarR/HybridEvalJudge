"""
User preferences API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID

from ..database import get_db
from ..models import User, UserPreference
from ..schemas import UserPreferenceResponse, UserPreferenceUpdate
from ..auth import get_current_user

router = APIRouter(prefix="/api/v1/preferences", tags=["preferences"])


@router.get("", response_model=UserPreferenceResponse)
async def get_user_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get current user's preferences.
    Creates default preferences if they don't exist.
    """
    # Check if preferences exist
    preferences = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id
    ).first()
    
    # Create default preferences if they don't exist
    if not preferences:
        preferences = UserPreference(
            user_id=current_user.id,
            default_judge_models=["gpt-4", "claude-3"],
            default_retrieval_enabled=True,
            default_aggregation_strategy="weighted_average",
            theme="light",
            notifications_enabled=True,
        )
        db.add(preferences)
        db.commit()
        db.refresh(preferences)
    
    return preferences


@router.put("", response_model=UserPreferenceResponse)
async def update_user_preferences(
    preferences_update: UserPreferenceUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update current user's preferences.
    Creates preferences if they don't exist.
    """
    # Get or create preferences
    preferences = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id
    ).first()
    
    if not preferences:
        # Create new preferences with provided values
        preferences = UserPreference(
            user_id=current_user.id,
            default_judge_models=preferences_update.default_judge_models or ["gpt-4", "claude-3"],
            default_retrieval_enabled=preferences_update.default_retrieval_enabled if preferences_update.default_retrieval_enabled is not None else True,
            default_aggregation_strategy=preferences_update.default_aggregation_strategy or "weighted_average",
            theme=preferences_update.theme or "light",
            notifications_enabled=preferences_update.notifications_enabled if preferences_update.notifications_enabled is not None else True,
        )
        db.add(preferences)
    else:
        # Update existing preferences
        update_data = preferences_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(preferences, field, value)
    
    db.commit()
    db.refresh(preferences)
    
    return preferences


@router.post("/reset", response_model=UserPreferenceResponse)
async def reset_user_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Reset current user's preferences to defaults.
    """
    # Get or create preferences
    preferences = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id
    ).first()
    
    if not preferences:
        preferences = UserPreference(user_id=current_user.id)
        db.add(preferences)
    else:
        # Reset to defaults
        preferences.default_judge_models = ["gpt-4", "claude-3"]
        preferences.default_retrieval_enabled = True
        preferences.default_aggregation_strategy = "weighted_average"
        preferences.theme = "light"
        preferences.notifications_enabled = True
    
    db.commit()
    db.refresh(preferences)
    
    return preferences


@router.delete("")
async def delete_user_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete current user's preferences.
    They will be recreated with defaults on next access.
    """
    preferences = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id
    ).first()
    
    if preferences:
        db.delete(preferences)
        db.commit()
    
    return {"message": "Preferences deleted successfully"}
