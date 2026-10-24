import hashlib
from typing import Any, Dict
from fastapi import HTTPException, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from itsdangerous import BadData, SignatureExpired, URLSafeTimedSerializer

from app.config import (
    GOOGLE_CLIENT_ID,
    SESSION_MAX_AGE,
    SESSION_SECRET,
    is_email_authorized,
)


def get_session_serializer() -> URLSafeTimedSerializer:
    """Instancia el serializador seguro configurado con SHA-256."""
    if not SESSION_SECRET:
        raise RuntimeError("SESSION_SECRET no está configurado en las variables de entorno.")
    return URLSafeTimedSerializer(
        secret_key=SESSION_SECRET,
        salt="dharma-session",
        signer_kwargs={"digest_method": hashlib.sha256},
    )


def verify_google_token(id_token_str: str) -> Dict[str, Any]:
    """
    Verifica el ID token emitido por Google:
    - Valida firma con certificados oficiales de Google
    - Valida que audience coincida con GOOGLE_CLIENT_ID
    - Valida email y email_verified
    - Valida que el email esté en AUTHORIZED_EMAILS
    """
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GOOGLE_CLIENT_ID no configurado en el servidor.",
        )

    try:
        request = google_requests.Request()
        id_info = id_token.verify_oauth2_token(
            id_token=id_token_str,
            request=request,
            audience=GOOGLE_CLIENT_ID,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de Google inválido o expirado.",
        ) from exc

    # Validar verificación de correo
    email = id_info.get("email")
    email_verified = id_info.get("email_verified", False)

    if not email or not email_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La cuenta de Google no tiene un correo verificado.",
        )

    # Validar lista blanca de correos autorizados
    if not is_email_authorized(email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"El correo {email} no está autorizado para acceder al sistema.",
        )

    return {
        "sub": id_info.get("sub"),
        "email": email.strip().lower(),
        "name": id_info.get("name", email),
        "picture": id_info.get("picture"),
    }


def create_session_token(payload: Dict[str, Any]) -> str:
    """
    Firma criptográficamente el payload de sesión con su timestamp.
    No cifra datos ni almacena secretos o tokens de Google.
    """
    serializer = get_session_serializer()
    safe_payload = {
        "sub": payload["sub"],
        "email": payload["email"],
        "name": payload["name"],
        "picture": payload.get("picture"),
    }
    return serializer.dumps(safe_payload)


def verify_session_token(token_str: str) -> Dict[str, Any]:
    """
    Verifica la firma y expiración de la sesión Dharma:
    - Comprueba integridad criptográfica (SHA-256)
    - Comprueba que no haya excedido el tiempo de vida (8 horas)
    - Valida que el correo continúe en la lista blanca de autorizados
    """
    serializer = get_session_serializer()
    try:
        payload = serializer.loads(token_str, max_age=SESSION_MAX_AGE)
    except SignatureExpired as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La sesión ha expirado.",
        ) from exc
    except BadData as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sesión inválida o adulterada.",
        ) from exc

    email = payload.get("email")
    if not email or not is_email_authorized(email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuario no autorizado.",
        )

    return payload
