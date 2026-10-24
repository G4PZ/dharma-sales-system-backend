from typing import Optional
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status

from app.config import SESSION_COOKIE_NAME, SESSION_MAX_AGE
from app.schemas.auth import GoogleLoginRequest, LogoutResponse, UserSession
from app.services import auth_service

router = APIRouter()


def get_current_user(
    dharma_session: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME)
) -> dict:
    """
    Dependencia de seguridad reutilizable:
    Extrae la cookie dharma_session y valida su firma, expiración y autorización.
    """
    if not dharma_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No se encontró una sesión activa.",
        )
    return auth_service.verify_session_token(dharma_session)


@router.post(
    "/google",
    response_model=UserSession,
    summary="Iniciar sesión con Google ID Token",
)
def login_with_google(body: GoogleLoginRequest, response: Response):
    """
    Verifica el ID token de Google, comprueba que el correo esté en la lista blanca,
    genera la sesión Dharma firmada y establece la cookie HttpOnly dharma_session.
    """
    user_data = auth_service.verify_google_token(body.id_token)
    session_token = auth_service.create_session_token(user_data)

    # Establecer la cookie de sesión Dharma
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=SESSION_MAX_AGE,
        path="/",
        httponly=True,
        samesite="lax",
        secure=False,  # En desarrollo sobre HTTP localhost
    )

    return user_data


@router.get(
    "/me",
    response_model=UserSession,
    summary="Obtener información del usuario autenticado",
)
def get_me(current_user: dict = Depends(get_current_user)):
    """
    Retorna la información del usuario autenticado a partir de su sesión válida.
    """
    return current_user


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Cerrar sesión en Dharma",
)
def logout(response: Response):
    """
    Destruye la cookie de sesión dharma_session con los mismos parámetros de path y seguridad.
    """
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
    )
    return {"message": "Sesión cerrada correctamente"}
