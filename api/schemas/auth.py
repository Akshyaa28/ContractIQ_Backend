from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class SignupRequest(BaseModel):
    """Register a new CMS or ACO user."""

    name: str = Field(..., min_length=2, max_length=100, description="Full name")
    email: str = Field(..., min_length=5, max_length=255, description="Email address")
    password: str = Field(..., min_length=6, max_length=128, description="Password (min 6 chars)")
    role: str = Field(..., pattern="^(CMS|ACO)$", description="'CMS' or 'ACO'")
    aco_name: Optional[str] = Field(None, max_length=255, description="ACO name (required for ACO role)")
    aco_id: Optional[str] = Field(None, max_length=50, description="ACO identifier (required for ACO role)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "CMS Admin",
                    "email": "cms@contractiq.com",
                    "password": "CMS@123",
                    "role": "CMS",
                },
                {
                    "name": "ACO Manager",
                    "email": "aco@contractiq.com",
                    "password": "ACO@123",
                    "role": "ACO",
                    "aco_name": "Golden Valley Care Alliance",
                    "aco_id": "A00001",
                },
            ]
        }
    }


class LoginRequest(BaseModel):
    """Login with email and password."""

    email: str = Field(..., description="Registered email address")
    password: str = Field(..., description="Account password")

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "cms@contractiq.com",
                "password": "CMS@123",
            }
        }
    }


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class UserResponse(BaseModel):
    """Public user profile (no password)."""

    id: str
    name: str
    email: str
    role: str
    aco_name: Optional[str] = None
    aco_id: Optional[str] = None
    is_active: bool

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """JWT token returned on successful login."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str
