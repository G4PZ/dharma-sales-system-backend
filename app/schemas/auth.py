from typing import Optional
from pydantic import BaseModel


class GoogleLoginRequest(BaseModel):
    id_token: str


class UserSession(BaseModel):
    sub: str
    email: str
    name: str
    picture: Optional[str] = None


class LogoutResponse(BaseModel):
    message: str
