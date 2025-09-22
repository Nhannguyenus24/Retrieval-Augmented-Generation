from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class NameRequest(BaseModel):
    name: str

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/hello")
def say_hello(request: NameRequest):
    return {"message": f"hello {request.name}"}
