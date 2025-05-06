from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.src.auth.dependencies import get_current_user, get_db

from .models import User as UserModel
from .schemas import User

router = APIRouter()


@router.get("/me", response_model=User)
def read_users_me(current_user: UserModel = Depends(get_current_user)):
    return current_user
