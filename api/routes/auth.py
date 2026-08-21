from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.models.database import get_db
from api.models.user import User
from api.schemas.auth import (
    SignupRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
    MessageResponse,
)
from api.services.auth_service import (
    create_user,
    authenticate_user,
    create_access_token,
)
from api.middleware.auth_middleware import get_current_user


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ============================================================
# SIGNUP
# ============================================================

@router.post(
    "/signup",
    response_model=TokenResponse,
    summary="Register a new user",
    description=(
        "Create a new CMS or ACO user account. "
        "Returns a JWT access token on success so the user "
        "is immediately logged in after registration."
    ),
    status_code=status.HTTP_201_CREATED,
)
def signup(
    request: SignupRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """
    Register a new user.

    - **role** must be 'CMS' or 'ACO'
    - ACO users must provide **aco_name** and **aco_id**
    - Password is hashed with bcrypt before storage
    - Returns JWT token + user profile immediately
    """

    try:
        user = create_user(
            db=db,
            name=request.name,
            email=request.email,
            password=request.password,
            role=request.role,
            aco_name=request.aco_name,
            aco_id=request.aco_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    # Generate token immediately (auto-login on signup)
    token = create_access_token(
        user_id=str(user.id),
        role=user.role,
    )

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=str(user.id),
            name=user.name,
            email=user.email,
            role=user.role,
            aco_name=user.aco_name,
            aco_id=user.aco_id,
            is_active=user.is_active,
        ),
    )


# ============================================================
# LOGIN
# ============================================================

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password",
    description=(
        "Authenticate with email and password. "
        "Returns a JWT access token valid for 24 hours."
    ),
    status_code=status.HTTP_200_OK,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """
    Authenticate a user.

    - Returns JWT token + user profile
    - Token is valid for 24 hours
    - Include token in subsequent requests as:
      `Authorization: Bearer <token>`
    """

    user = authenticate_user(
        db=db,
        email=request.email,
        password=request.password,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_access_token(
        user_id=str(user.id),
        role=user.role,
    )

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=str(user.id),
            name=user.name,
            email=user.email,
            role=user.role,
            aco_name=user.aco_name,
            aco_id=user.aco_id,
            is_active=user.is_active,
        ),
    )


# ============================================================
# GET CURRENT USER (ME)
# ============================================================

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    description="Returns the profile of the currently authenticated user.",
    status_code=status.HTTP_200_OK,
)
def get_me(
    user: User = Depends(get_current_user),
) -> UserResponse:
    """
    Returns the currently authenticated user's profile.

    Requires a valid JWT token in the Authorization header.
    """

    return UserResponse(
        id=str(user.id),
        name=user.name,
        email=user.email,
        role=user.role,
        aco_name=user.aco_name,
        aco_id=user.aco_id,
        is_active=user.is_active,
    )


# ============================================================
# LOGOUT (client-side — just for documentation)
# ============================================================

@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout (client-side)",
    description=(
        "JWT tokens are stateless — the client discards the token to logout. "
        "This endpoint exists for documentation and frontend consistency."
    ),
    status_code=status.HTTP_200_OK,
)
def logout(
    user: User = Depends(get_current_user),
) -> MessageResponse:
    """
    Logout the current user.

    Since JWT is stateless, the actual logout happens client-side
    by discarding the token. This endpoint confirms the token was
    valid at the time of logout.
    """
    return MessageResponse(message=f"User {user.email} logged out successfully.")
