@echo off
echo Starting AI News Monitor System...

echo.
echo ==============================================
echo 1. Creating necessary directories...
echo ==============================================
if not exist data mkdir data
if not exist models mkdir models
if not exist logs mkdir logs
echo Directories confirmed.

echo.
echo ==============================================
echo 2. Running Data Ingestion and Processing...
echo ==============================================
start "Data Ingestion" cmd /k "python -m src.ingestion.fetcher"

echo.
echo ==============================================
echo 2b. Running Intelligence Pipeline...
echo ==============================================
start "Intelligence Pipeline" cmd /k "python -m src.intelligence.pipeline"

echo.
echo ==============================================
echo 3. Starting API Backend (Uvicorn)...
echo ==============================================
start "API Backend" cmd /k "python -m uvicorn src.api.main:app --reload --port 8000"

echo.
echo ==============================================
echo 4. Starting Frontend Dashboard (Vite)...
echo ==============================================
start "Frontend Dashboard" cmd /k "cd dashboard && npm run dev"

echo.
echo ==============================================
echo 5. Starting Background Scheduler...
echo ==============================================
start "Background Scheduler" cmd /k "python -m src.ingestion.scheduler"

echo.
echo ==============================================
echo 6. Starting Reprocessor...
echo ==============================================
start "Reprocessor" cmd /k "python -m src.maintenance.reprocess_fakes"

echo.
echo ==============================================
echo AI News Monitor System Startup Initiated!
echo ==============================================
echo.
echo Please wait a moment for all services to initialize.
echo The dashboard will be available at: http://localhost:5173
echo The API will be available at: http://localhost:8000
echo.
echo You can safely close this orchestrator window. 
echo Do NOT close the individual command prompt windows that opened.
echo.
pause
