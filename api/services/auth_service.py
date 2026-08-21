import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from api.models.user import User

# ============================================================
# CONFIGURATION
# ============================================================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "contractiq-dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ============================================================
# JWT TOKENS
# ============================================================

def create_access_token(user_id: str, role: str) -> str:
    """
    Create a JWT access token.

    Payload contains:
      - sub: user_id (UUID as string)
      - role: 'CMS' or 'ACO'
      - exp: expiration timestamp
    """
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token.
    Returns the payload dict or None if invalid/expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ============================================================
# USER CRUD
# ============================================================

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Find a user by email (case-insensitive)."""
    return db.query(User).filter(
        User.email == email.lower().strip()
    ).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    """Find a user by UUID."""
    return db.query(User).filter(User.id == user_id).first()


def create_user(
    db: Session,
    name: str,
    email: str,
    password: str,
    role: str,
    aco_name: Optional[str] = None,
    aco_id: Optional[str] = None,
) -> User:
    """
    Create a new user in the database.
    Raises ValueError if email already exists.
    """
    # Check for existing email
    existing = get_user_by_email(db, email)
    if existing:
        raise ValueError("An account with this email already exists.")

    # Validate ACO fields
    if role == "ACO" and (not aco_name or not aco_id):
        raise ValueError("ACO users must provide aco_name and aco_id.")

    user = User(
        name=name.strip(),
        email=email.lower().strip(),
        password_hash=hash_password(password),
        role=role,
        aco_name=aco_name.strip() if aco_name else None,
        aco_id=aco_id.strip() if aco_id else None,
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Authenticate by email + password.
    Returns the User object if valid, None otherwise.
    """
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
