# backend/app/api/__init__.py
from fastapi import APIRouter

api_router = APIRouter()

from .endpoints import *  # Import all endpoints from the endpoints module