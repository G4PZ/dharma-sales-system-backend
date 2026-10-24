import json
from typing import Any, Dict, Optional
import httpx

from app.config import (
    SUNAT_INTERNAL_TOKEN,
    SUNAT_REQUEST_TIMEOUT,
    SUNAT_SERVICE_URL,
)


class SunatClientError(Exception):
    """Clase base para errores de comunicación con el microservicio SUNAT."""

    def __init__(self, message: str, code: str = "ERROR_ENVIO", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class SunatConnectionError(SunatClientError):
    """Ocurre cuando Apache está apagado, timeout o conexión rechazada."""

    def __init__(self, message: str = "Microservicio SUNAT no disponible o fuera de línea."):
        super().__init__(message=message, code="ERROR_ENVIO", status_code=503)


class SunatAuthError(SunatClientError):
    """Ocurre cuando PHP rechaza el token interno (401)."""

    def __init__(self, message: str = "Error de autenticación interna con el microservicio SUNAT."):
        super().__init__(message=message, code="AUTH_ERROR", status_code=500)


class SunatPayloadError(SunatClientError):
    """Ocurre cuando PHP responde 400 con error controlado de negocio/formato."""

    def __init__(self, message: str, code: str = "INVALID_PAYLOAD"):
        super().__init__(message=message, code=code, status_code=400)


class SunatClient:
    """Cliente HTTP centralizado para interactuar con el microservicio PHP/Greenter."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        internal_token: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or SUNAT_SERVICE_URL).rstrip("/")
        self.internal_token = internal_token or SUNAT_INTERNAL_TOKEN
        self.timeout = timeout or SUNAT_REQUEST_TIMEOUT

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Internal-Token": self.internal_token,
        }

    def _send_request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=self._get_headers(),
                    json=json_data,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as exc:
            raise SunatConnectionError(
                f"No se pudo establecer conexión con el microservicio SUNAT en '{self.base_url}'. Verifique que Apache/XAMPP esté encendido."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise SunatConnectionError(
                f"Tiempo de espera agotado ({self.timeout}s) esperando respuesta del microservicio SUNAT."
            ) from exc
        except Exception as exc:
            raise SunatConnectionError(f"Error de red al conectar con el microservicio SUNAT: {str(exc)}") from exc

        # Procesar código de respuesta HTTP
        if response.status_code == 401:
            raise SunatAuthError("Token de acceso interno no proporcionado o inválido entre FastAPI y PHP.")

        try:
            data = response.json()
        except json.JSONDecodeError:
            if response.is_error:
                raise SunatClientError(
                    f"Respuesta no válida del microservicio SUNAT (HTTP {response.status_code}): {response.text[:200]}"
                )
            return {"raw": response.text}

        if response.status_code == 400:
            error_code = data.get("error") or data.get("code") or "INVALID_PAYLOAD"
            error_msg = data.get("message") or data.get("descripcion") or "Error en el payload del comprobante."
            raise SunatPayloadError(message=error_msg, code=error_code)

        if response.status_code >= 500:
            error_code = data.get("error") or data.get("code") or "SYSTEM_ERROR"
            error_msg = data.get("message") or "Error interno en el microservicio SUNAT."
            raise SunatClientError(message=error_msg, code=error_code, status_code=500)

        return data

    def check_health(self) -> Dict[str, Any]:
        """Consulta el estado del microservicio PHP y sus extensiones."""
        return self._send_request("GET", "health.php")

    def emitir_factura(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Envía el payload de una Factura (01) a emitir.php."""
        return self._send_request("POST", "api/emitir.php", json_data=payload)

    def emitir_resumen(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Envía el payload de un Resumen Diario de Boletas a api/resumen.php."""
        return self._send_request("POST", "api/resumen.php", json_data=payload)

    def consultar_ticket(self, ticket: str) -> Dict[str, Any]:
        """Consulta el estado de un ticket ante SUNAT mediante api/consultar.php."""
        return self._send_request("POST", "api/consultar.php", json_data={"ticket": ticket})


# Instancia singleton predeterminada
sunat_client = SunatClient()
