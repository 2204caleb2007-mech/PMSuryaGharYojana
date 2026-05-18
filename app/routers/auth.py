"""
Authentication routes — JWT-based (local) + Google OAuth (ID token verification)
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

# Google token verification
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from app.database import get_db
from app.models import User, AuthProvider
from app.schemas import UserRegister, UserLogin, Token, UserResponse, GoogleAuthRequest
from app.core.security import verify_password, get_password_hash, create_access_token
from app.core.config import settings
from app.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ── helpers ─────────────────────────────────────────────────────────────────────

def _make_token(user: User) -> Token:
    """Create a JWT access token for the given user and wrap it in a Token response."""
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=access_token, token_type="bearer", user=user)


# ── Local Auth ───────────────────────────────────────────────────────────────────

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new local user and return JWT token immediately."""
    existing_user = db.query(User).filter(
        (User.email == user_data.email) | (User.username == user_data.username)
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or username already registered"
        )

    db_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        auth_provider=AuthProvider.LOCAL,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return _make_token(db_user)


@router.post("/login", response_model=Token)
async def login(login_data: UserLogin, db: Session = Depends(get_db)):
    """Login with email + password and return a JWT access token."""
    user = db.query(User).filter(User.email == login_data.email).first()

    if not user or not user.hashed_password:
        # user exists but signed up via Google — no password set
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return _make_token(user)


# ── Google OAuth ─────────────────────────────────────────────────────────────────

@router.post("/google", response_model=Token)
async def google_auth(body: GoogleAuthRequest, db: Session = Depends(get_db)):
    """
    Verify a Google ID token issued by the GSI popup on the frontend.

    Flow:
      1. Frontend calls google.accounts.id.initialize({ … }) — user clicks button.
      2. Google returns a credential (signed JWT) to our callback.
      3. Frontend POSTs that credential here.
      4. We verify it with google-auth library (checks aud, iss, exp, signature).
      5. We upsert the user (create if new, update picture if existing).
      6. We return our own JWT so the rest of the app works identically.
    """
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured on this server."
        )

    # ── 1. Verify the Google ID token ────────────────────────────────────────
    try:
        id_info = id_token.verify_oauth2_token(
            body.credential,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,   # validates the `aud` claim
        )
    except ValueError as exc:
        logger.warning(f"Google token verification failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token. Please try signing in again.",
        )

    # Google guarantees these fields exist after a successful verify
    google_id: str = id_info["sub"]           # stable unique Google user ID
    email: str     = id_info["email"]
    full_name: str = id_info.get("name", "")
    picture: str   = id_info.get("picture", "")
    email_verified: bool = id_info.get("email_verified", False)

    if not email_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your Google account email is not verified.",
        )

    # ── 2. Upsert user ───────────────────────────────────────────────────────
    user = (
        db.query(User)
        .filter((User.google_id == google_id) | (User.email == email))
        .first()
    )

    if user:
        # Existing user — link Google account if not already linked
        if not user.google_id:
            user.google_id = google_id
            user.auth_provider = AuthProvider.GOOGLE
        # Refresh avatar on every login (picture URL can change)
        if picture:
            user.profile_picture = picture
        if not user.full_name and full_name:
            user.full_name = full_name
        db.commit()
        db.refresh(user)
    else:
        # New user — create account (no password, no username needed)
        user = User(
            email=email,
            full_name=full_name,
            google_id=google_id,
            profile_picture=picture,
            auth_provider=AuthProvider.GOOGLE,
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    logger.info(f"Google auth success: user_id={user.id}, email={user.email}")
    return _make_token(user)


# ── Shared ───────────────────────────────────────────────────────────────────────

@router.post("/logout")
async def logout():
    """Logout — client must drop the token from storage."""
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    return current_user
