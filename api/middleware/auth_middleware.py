from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from api.models.database import get_db
from api.models.user import User
from api.services.auth_service import decode_access_token, get_user_by_id

# ============================================================
# BEARER TOKEN EXTRACTION
# ============================================================

security = HTTPBearer()


# ============================================================
# GET CURRENT USER (dependency)
# ============================================================

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency that extracts and validates the JWT token
    from the Authorization header, then returns the User object.

    Usage in routes:
        @router.get("/protected")
        def my_endpoint(user: User = Depends(get_current_user)):
            ...
    """
    token = credentials.credentials

    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    return user


# ============================================================
# ROLE-BASED ACCESS (dependency factory)
# ============================================================

def require_role(required_role: str):
    """
    Returns a dependency that checks the user's role.

    Usage:
        @router.get("/cms-only", dependencies=[Depends(require_role("CMS"))])
        def cms_endpoint():
            ...
    """

    def role_checker(user: User = Depends(get_current_user)) -> User:
        if user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. This endpoint requires '{required_role}' role.",
            )
        return user

    return role_checker
