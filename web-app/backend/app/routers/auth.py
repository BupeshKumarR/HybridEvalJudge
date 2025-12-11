"""
Authentication router for user registration, login, and logout.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging

from ..database import get_db
from ..models import User, UserPreference
from ..schemas import (
    UserCreate,
    UserResponse,
    Token,
    LoginRequest
)
from ..auth import (
    get_password_hash,
    authenticate_user,
    create_access_token,
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from ..security import input_sanitizer
from ..audit_log import audit_logger, AuditEventType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: Request,
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user account.
    
    Args:
        request: FastAPI request object
        user_data: User registration data (username, email, password)
        db: Database session
        
    Returns:
        Created user object
        
    Raises:
        HTTPException: If username or email already exists
    """
    # Get client IP for audit logging
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    user_agent = request.headers.get("User-Agent", "")
    
    # Sanitize inputs
    username = input_sanitizer.sanitize_text(user_data.username, max_length=255)
    email = input_sanitizer.sanitize_text(user_data.email, max_length=255)
    
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        # Log failed registration attempt
        audit_logger.log_authentication_event(
            event_type=AuditEventType.REGISTRATION,
            username=username,
            ip_address=client_ip,
            user_agent=user_agent,
            success=False,
            details={"reason": "username_exists"}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = db.query(User).filter(User.email == email).first()
    if existing_email:
        # Log failed registration attempt
        audit_logger.log_authentication_event(
            event_type=AuditEventType.REGISTRATION,
            username=username,
            ip_address=client_ip,
            user_agent=user_agent,
            success=False,
            details={"reason": "email_exists"}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=username,
        email=email,
        password_hash=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create default user preferences
    default_preferences = UserPreference(
        user_id=new_user.id,
        default_judge_models=["gpt-4", "claude-3"],
        default_retrieval_enabled=True,
        default_aggregation_strategy="weighted_average",
        theme="light",
        notifications_enabled=True
    )
    
    db.add(default_preferences)
    db.commit()
    
    logger.info(f"New user registered: {new_user.username} (ID: {new_user.id})")
    
    # Log successful registration
    audit_logger.log_authentication_event(
        event_type=AuditEventType.REGISTRATION,
        username=username,
        ip_address=client_ip,
        user_agent=user_agent,
        success=True,
        details={"user_id": str(new_user.id)}
    )
    
    return new_user


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login with username and password to get access token.
    
    Args:
        request: FastAPI request object
        form_data: OAuth2 form data with username and password
        db: Database session
        
    Returns:
        Access token
        
    Raises:
        HTTPException: If authentication fails
    """
    # Get client IP for audit logging
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    user_agent = request.headers.get("User-Agent", "")
    
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        # Log failed login attempt
        audit_logger.log_authentication_event(
            event_type=AuditEventType.LOGIN_FAILURE,
            username=form_data.username,
            ip_address=client_ip,
            user_agent=user_agent,
            success=False,
            details={"reason": "invalid_credentials"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login timestamp
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.username} (ID: {user.id})")
    
    # Log successful login
    audit_logger.log_authentication_event(
        event_type=AuditEventType.LOGIN_SUCCESS,
        username=user.username,
        ip_address=client_ip,
        user_agent=user_agent,
        success=True,
        details={"user_id": str(user.id)}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login/json", response_model=Token)
async def login_json(
    request: Request,
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    Login with JSON body (alternative to form data).
    
    Args:
        request: FastAPI request object
        login_data: Login credentials
        db: Database session
        
    Returns:
        Access token
        
    Raises:
        HTTPException: If authentication fails
    """
    # Get client IP for audit logging
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    user_agent = request.headers.get("User-Agent", "")
    
    user = authenticate_user(db, login_data.username, login_data.password)
    if not user:
        # Log failed login attempt
        audit_logger.log_authentication_event(
            event_type=AuditEventType.LOGIN_FAILURE,
            username=login_data.username,
            ip_address=client_ip,
            user_agent=user_agent,
            success=False,
            details={"reason": "invalid_credentials"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login timestamp
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.username} (ID: {user.id})")
    
    # Log successful login
    audit_logger.log_authentication_event(
        event_type=AuditEventType.LOGIN_SUCCESS,
        username=user.username,
        ip_address=client_ip,
        user_agent=user_agent,
        success=True,
        details={"user_id": str(user.id)}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """
    Logout current user.
    
    Note: Since we're using stateless JWT tokens, logout is handled client-side
    by removing the token. This endpoint is provided for consistency and can be
    extended to implement token blacklisting if needed.
    
    Args:
        request: FastAPI request object
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    # Get client IP for audit logging
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    user_agent = request.headers.get("User-Agent", "")
    
    logger.info(f"User logged out: {current_user.username} (ID: {current_user.id})")
    
    # Log logout event
    audit_logger.log_authentication_event(
        event_type=AuditEventType.LOGOUT,
        username=current_user.username,
        ip_address=client_ip,
        user_agent=user_agent,
        success=True,
        details={"user_id": str(current_user.id)}
    )
    
    return {
        "message": "Successfully logged out",
        "detail": "Please remove the access token from client storage"
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
    """
    return current_user


@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_active_user)
):
    """
    Refresh access token for current user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        New access token
    """
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(current_user.id), "username": current_user.username},
        expires_delta=access_token_expires
    )
    
    logger.info(f"Token refreshed for user: {current_user.username} (ID: {current_user.id})")
    
    return {"access_token": access_token, "token_type": "bearer"}
