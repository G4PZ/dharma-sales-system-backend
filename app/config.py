import os
from typing import List, Set
from dotenv import load_dotenv

load_dotenv()

GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
SESSION_SECRET: str = os.getenv("SESSION_SECRET", "")
AUTHORIZED_EMAILS_RAW: str = os.getenv("AUTHORIZED_EMAILS", "")

# Configuración de cookies y sesión
SESSION_COOKIE_NAME: str = "dharma_session"
SESSION_MAX_AGE: int = 8 * 3600  # 8 horas en segundos

# Configuración de Microservicio SUNAT / Greenter (Fase 4)
SUNAT_SERVICE_URL: str = os.getenv("SUNAT_SERVICE_URL", "http://localhost/dharma-sunat")
SUNAT_INTERNAL_TOKEN: str = os.getenv("SUNAT_INTERNAL_TOKEN", "")
SUNAT_REQUEST_TIMEOUT: int = int(os.getenv("SUNAT_REQUEST_TIMEOUT", "30"))

# Configuración del Módulo de Inteligencia Artificial
DEFAULT_ML_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
ML_MODELS_DIR: str = os.getenv("ML_MODELS_DIR", DEFAULT_ML_MODELS_DIR)


def get_authorized_emails() -> Set[str]:
    """Retorna el conjunto de correos autorizados en minúsculas y sin espacios."""
    if not AUTHORIZED_EMAILS_RAW:
        return set()
    return {
        email.strip().lower()
        for email in AUTHORIZED_EMAILS_RAW.split(",")
        if email.strip()
    }

def is_email_authorized(email: str) -> bool:
    """Verifica si el correo está en la lista de autorizados."""
    if not email:
        return False
    return email.strip().lower() in get_authorized_emails()
