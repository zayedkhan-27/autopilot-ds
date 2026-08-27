# AutoPilot-DS Production Website

This folder upgrades your Streamlit ML project into a full-stack application.

## Architecture

- `backend/engine` — your original ML engine (preserved)
- `backend/app` — FastAPI API, authentication, database and file management
- `frontend` — Next.js premium user interface
- PostgreSQL in production, SQLite for local development

## Run locally

### Backend
```bash
cd backend
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open:
- Frontend: http://localhost:3000
- Backend docs: http://localhost:8000/docs

## Important environment variables

Copy `.env.example` to `.env`.

For local testing, the default SQLite database works automatically.
For production, set `DATABASE_URL` to a PostgreSQL URL.
