import os
import shutil
from pathlib import Path
from fastapi import (FastAPI, Depends, HTTPException, UploadFile, File)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import pandas as pd

from .config import settings
from .database import Base, engine, get_db
from .models import User, Dataset, Experiment
from .schemas import RegisterRequest, LoginRequest, TokenResponse, ExperimentRequest
from .security import hash_password, verify_password, create_access_token, get_current_user
from .services import run_experiment

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AutoPilot DS API", version="1.0.0")
origins = [x.strip() for x in settings.cors_origins.split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

STORAGE = Path(__file__).resolve().parents[1] / "storage" / "uploads"
STORAGE.mkdir(parents=True, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok", "service": "AutoPilot DS API"}

@app.post("/api/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email.lower()).first():
        raise HTTPException(409, "Email already registered")
    user = User(name=payload.name, email=payload.email.lower(),
                password_hash=hash_password(payload.password))
    db.add(user); db.commit(); db.refresh(user)
    return {"access_token": create_access_token(user.id)}

@app.post("/api/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    return {"access_token": create_access_token(user.id)}
@app.post("/api/auth/token", response_model=TokenResponse)
def token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == form_data.username.lower())
        .first()
    )

    if not user or not verify_password(
        form_data.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    return {
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
    }

@app.get("/api/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "name": user.name, "email": user.email}

@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...), user: User = Depends(get_current_user),
                         db: Session = Depends(get_db)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".csv", ".xlsx"}:
        raise HTTPException(400, "Only CSV and XLSX files are supported")
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
    filename = f"{user.id}_{os.urandom(8).hex()}{ext}"
    path = STORAGE / filename
    path.write_bytes(content)
    try:
        df = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)
        if df.empty or len(df.columns) == 0:
            raise ValueError("Dataset is empty")
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read dataset: {exc}")
    ds = Dataset(user_id=user.id, name=Path(file.filename).stem,
                 original_filename=file.filename, file_path=str(path),
                 rows=len(df), columns=len(df.columns))
    db.add(ds); db.commit(); db.refresh(ds)
    return {"id": ds.id, "name": ds.name, "rows": ds.rows, "columns": ds.columns}

@app.get("/api/datasets")
def list_datasets(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Dataset).filter(Dataset.user_id == user.id).order_by(Dataset.id.desc()).all()
    return [{"id": x.id, "name": x.name, "original_filename": x.original_filename,
             "rows": x.rows, "columns": x.columns, "created_at": x.created_at} for x in rows]

@app.post("/api/experiments")
def create_experiment(payload: ExperimentRequest, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    ds = db.get(Dataset, payload.dataset_id)
    if not ds or ds.user_id != user.id:
        raise HTTPException(404, "Dataset not found")
    exp = Experiment(user_id=user.id, dataset_id=ds.id, status="QUEUED",
                     target_column=payload.target_column, task_type=payload.task_type,
                     max_iterations=payload.max_iterations)
    db.add(exp); db.commit(); db.refresh(exp)
    run_experiment(db, exp, ds)
    return {"id": exp.id, "status": exp.status, "result": exp.result_json}

@app.get("/api/experiments")
def list_experiments(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Experiment).filter(Experiment.user_id == user.id).order_by(Experiment.id.desc()).all()
    return [{"id": x.id, "dataset_id": x.dataset_id, "status": x.status,
             "task_type": x.task_type, "target_column": x.target_column,
             "best_model_name": x.best_model_name, "health_score": x.health_score,
             "created_at": x.created_at} for x in rows]

@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: int, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    exp = db.get(Experiment, experiment_id)
    if not exp or exp.user_id != user.id:
        raise HTTPException(404, "Experiment not found")
    return {"id": exp.id, "status": exp.status, "error_message": exp.error_message,
            "result": exp.result_json}
