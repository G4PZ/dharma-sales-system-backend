from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.client import (
    ClientCreate,
    ClientListResponse,
    ClientResponse,
    ClientStatsResponse,
    ClientUpdate,
)
from app.routes.auth import get_current_user
from app.services import client_service

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("", response_model=ClientListResponse, summary="Listar clientes con filtros y paginación")
def list_clients(
    search: Optional[str] = Query(None, description="Búsqueda por RUC, razón social, contacto o correo"),
    estado: Optional[str] = Query(None, description="Filtrar por estado: 'activo', 'inactivo', 'todos'"),
    skip: int = Query(0, ge=0, description="Cantidad de registros a omitir"),
    limit: int = Query(10, ge=1, le=100, description="Cantidad de registros por página"),
    db: Session = Depends(get_db),
):
    return client_service.get_clients(
        db=db,
        search=search,
        estado=estado,
        skip=skip,
        limit=limit,
    )


@router.get("/stats", response_model=ClientStatsResponse, summary="Obtener métricas de clientes")
def get_client_stats(db: Session = Depends(get_db)):
    return client_service.get_client_stats(db=db)


@router.get("/{cliente_id}", response_model=ClientResponse, summary="Obtener cliente por ID")
def get_client(cliente_id: int, db: Session = Depends(get_db)):
    client = client_service.get_client_by_id(db=db, client_id=cliente_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cliente con ID {cliente_id} no encontrado."
        )
    return client


@router.post("", response_model=ClientResponse, status_code=status.HTTP_201_CREATED, summary="Registrar nuevo cliente")
def create_client(client_in: ClientCreate, db: Session = Depends(get_db)):
    return client_service.create_client(db=db, client_in=client_in)


@router.put("/{cliente_id}", response_model=ClientResponse, summary="Actualizar información de un cliente")
def update_client(
    cliente_id: int,
    client_in: ClientUpdate,
    db: Session = Depends(get_db),
):
    client = client_service.get_client_by_id(db=db, client_id=cliente_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cliente con ID {cliente_id} no encontrado."
        )
    return client_service.update_client(db=db, db_client=client, client_in=client_in)


@router.patch("/{cliente_id}/toggle-status", response_model=ClientResponse, summary="Activar o desactivar cliente")
def toggle_client_status(cliente_id: int, db: Session = Depends(get_db)):
    client = client_service.get_client_by_id(db=db, client_id=cliente_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cliente con ID {cliente_id} no encontrado."
        )
    return client_service.toggle_client_status(db=db, db_client=client)
