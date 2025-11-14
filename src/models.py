from pydantic import BaseModel, Field
from typing import List


class QuestionRequest(BaseModel):
    question: str


class DocumentsResponse(BaseModel):
    message: str
    documents_indexed: int
    total_chunks: int


class QuestionResponse(BaseModel):
    answer: str
    references: List[str]


class RegisterRequest(BaseModel):
    user_name: str
    senha: str


class LoginRequest(BaseModel):
    user_name: str
    senha: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
