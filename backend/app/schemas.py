from pydantic import BaseModel, EmailStr, Field

class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ExperimentRequest(BaseModel):
    dataset_id: int
    target_column: str | None = None
    task_type: str | None = None
    max_iterations: int = Field(default=5, ge=1, le=8)
