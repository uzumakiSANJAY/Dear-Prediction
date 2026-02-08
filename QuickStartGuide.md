# Lottery Prediction - Quick Start Guide

## Architecture Overview

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│  Frontend   │────>│  Backend    │────>│ ML Service  │     │ PostgreSQL │
│  React+Vite │     │  Express.js │     │  FastAPI    │     │  Database  │
│  Port 3000  │     │  Port 5000  │────>│  Port 8001  │     │  Port 5432 │
└────────────┘     └─────┬───────┘     └─────┬───────┘     └─────┬──────┘
                         │                   │                    │
                         └───────────────────┴────────────────────┘
                                    PostgreSQL Connection
```

---

## Prerequisites

- **Node.js** >= 18 (recommended: 20 LTS)
- **Python** >= 3.10 (recommended: 3.11)
- **PostgreSQL** >= 15 (recommended: 16)
- **npm** (comes with Node.js)
- **pip** (comes with Python)

---

## 1. Database Setup

### Option A: Local PostgreSQL

1. Install PostgreSQL on your machine.
2. Create the database and user:

```sql
CREATE USER lottery_user WITH PASSWORD 'lottery_pass_2024';
CREATE DATABASE lottery_db OWNER lottery_user;
GRANT ALL PRIVILEGES ON DATABASE lottery_db TO lottery_user;
```

### Option B: Docker (PostgreSQL only)

Run just the database container:

```bash
docker run -d \
  --name lottery-postgres \
  -e POSTGRES_USER=lottery_user \
  -e POSTGRES_PASSWORD=lottery_pass_2024 \
  -e POSTGRES_DB=lottery_db \
  -p 5432:5432 \
  postgres:16-alpine
```

---

## 2. Environment Files

Each service has its own `.env` file for local development. These are already created with sensible defaults.

### backend/.env

```env
POSTGRES_USER=lottery_user
POSTGRES_PASSWORD=lottery_pass_2024
POSTGRES_DB=lottery_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
BACKEND_PORT=5000
NODE_ENV=development
ML_SERVICE_URL=http://localhost:8001
LOTTERY_API_BASE=https://indialotteryapi.com/wp-json/dearlottery/v1
```

### frontend/.env

```env
VITE_API_URL=http://localhost:5000/api
```

### ml-service/.env

```env
POSTGRES_USER=lottery_user
POSTGRES_PASSWORD=lottery_pass_2024
POSTGRES_DB=lottery_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

> Adjust credentials/ports as needed for your local PostgreSQL setup.

---

## 3. Running Each Service

Open **3 separate terminals** and run each service independently.

### Terminal 1 - ML Service (Python)

```bash
cd ml-service

# Create virtual environment (first time only)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies (first time or after requirements change)
pip install -r requirements.txt

# Start the ML service
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

ML service will be running at: **http://localhost:8001**
API docs available at: **http://localhost:8001/docs**

### Terminal 2 - Backend (Node.js)

```bash
cd backend

# Install dependencies (first time or after package.json change)
npm install

# Start the backend
npm run dev
```

Backend will be running at: **http://localhost:5000**
Health check: **http://localhost:5000/api/health**

### Terminal 3 - Frontend (React)

```bash
cd frontend

# Install dependencies (first time or after package.json change)
npm install

# Start the frontend
npm run dev
```

Frontend will be running at: **http://localhost:3000**

---

## 4. Startup Order (Recommended)

Start services in this order for best results:

1. **PostgreSQL** - Database must be running first
2. **ML Service** - Backend calls ML service on startup
3. **Backend** - Connects to both PostgreSQL and ML service
4. **Frontend** - Connects to backend API

> The backend will auto-initialize the database schema and begin fetching historical lottery data on first run.

---

## 5. Verifying Everything Works

### Check health endpoints:

```bash
# ML Service
curl http://localhost:8001/health

# Backend
curl http://localhost:5000/api/health
```

### Open the frontend:

Navigate to **http://localhost:3000** in your browser. You should see:
- A dashboard with connection status (green = connected)
- Prediction cards for 1pm, 6pm, and 8pm slots
- Buttons to Sync Data and Generate Predictions

---

## 6. Key API Endpoints

### Backend (port 5000)

| Method | Endpoint                    | Description                 |
|--------|-----------------------------|-----------------------------|
| GET    | `/api/health`               | Health check                |
| GET    | `/api/lottery/results`      | Get lottery results         |
| GET    | `/api/lottery/results/latest` | Latest result per slot    |
| GET    | `/api/lottery/stats`        | Database statistics         |
| POST   | `/api/lottery/sync`         | Fetch latest lottery data   |
| GET    | `/api/predictions/latest`   | Latest predictions          |
| POST   | `/api/predictions/generate` | Generate new predictions    |
| POST   | `/api/predictions/train`    | Retrain ML models           |

### ML Service (port 8001)

| Method | Endpoint    | Description             |
|--------|-------------|-------------------------|
| GET    | `/health`   | Health check            |
| POST   | `/predict`  | Generate predictions    |
| POST   | `/train`    | Train models on data    |

---

## 7. Alternative: Docker Compose (All Services)

To run everything with Docker instead:

```bash
# From project root
docker-compose up --build
```

This starts all services including PostgreSQL and Adminer (DB admin UI at port 8080).

---

## Troubleshooting

### Backend fails to connect to PostgreSQL
- Ensure PostgreSQL is running on `localhost:5432`
- Verify credentials in `backend/.env` match your PostgreSQL setup
- Check that the `lottery_db` database exists

### Backend fails to connect to ML Service
- Ensure the ML service is running on port 8001
- Check `ML_SERVICE_URL` in `backend/.env` is `http://localhost:8001`

### Frontend shows "disconnected" status
- Ensure the backend is running on port 5000
- Check browser console for CORS errors
- Verify `VITE_API_URL` in `frontend/.env` is `http://localhost:5000/api`

### ML Service import errors
- Make sure you're running from the `ml-service/` directory
- Ensure the virtual environment is activated
- Run `pip install -r requirements.txt` to install all dependencies

### Port conflicts
If default ports are in use, update the respective `.env` files:
- Backend: change `BACKEND_PORT` in `backend/.env`
- Frontend: change the `port` in `frontend/vite.config.js`
- ML Service: change the `--port` flag in the uvicorn command
- Update `ML_SERVICE_URL` in `backend/.env` if ML port changes
- Update `VITE_API_URL` in `frontend/.env` if backend port changes


 # Terminal 1 - ML Service
  cd ml-service
  python -m venv venv && venv\Scripts\activate
  pip install -r requirements.txt
  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

  # Terminal 2 - Backend
  cd backend
  npm install
  npm run dev

  # Terminal 3 - Frontend
  cd frontend
  npm install
  npm run dev
