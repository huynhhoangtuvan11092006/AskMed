<#
.SYNOPSIS
    Ask_Med Windows PowerShell Management Script
.DESCRIPTION
    Convenient CLI manager mirroring Makefile targets for Windows environments.
.EXAMPLE
    .\scripts\manage.ps1 docker-up
    .\scripts\manage.ps1 ui
#>

param(
    [Parameter(Position=0)]
    [string]$Action = "help"
)

$ErrorActionPreference = "Stop"

switch ($Action.ToLower()) {
    "install" {
        Write-Host "--> Installing project dependencies..." -ForegroundColor Cyan
        & ".\venv\Scripts\python.exe" -m pip install --upgrade pip
        & ".\venv\Scripts\python.exe" -m pip install -r requirements.txt
    }
    "docker-build" {
        Write-Host "--> Building Ask_Med Docker images..." -ForegroundColor Cyan
        docker compose build
    }
    "docker-up" {
        Write-Host "--> Starting Ask_Med containers (Ollama, API, UI)..." -ForegroundColor Cyan
        docker compose up -d
        Write-Host "--> Services running: API on http://localhost:8000 | UI on http://localhost:8501" -ForegroundColor Green
    }
    "docker-down" {
        Write-Host "--> Stopping Ask_Med containers..." -ForegroundColor Cyan
        docker compose down
    }
    "docker-logs" {
        docker compose logs -f
    }
    "docker-gpu" {
        Write-Host "--> Launching Ask_Med with NVIDIA GPU acceleration..." -ForegroundColor Cyan
        docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
    }
    "api" {
        Write-Host "--> Starting FastAPI backend locally (port 8000)..." -ForegroundColor Cyan
        & ".\venv\Scripts\python.exe" -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
    }
    "ui" {
        Write-Host "--> Starting Streamlit UI locally (port 8501)..." -ForegroundColor Cyan
        & ".\venv\Scripts\python.exe" -m streamlit run src/ui/app.py
    }
    "ingest" {
        Write-Host "--> Executing PubMed Ingestion Pipeline..." -ForegroundColor Cyan
        & ".\venv\Scripts\python.exe" -m src.ingestion.ingest_pipeline --query "diabetes mellitus treatment" --limit 10
    }
    "eval" {
        Write-Host "--> Running 25-Question RAGAs Benchmark Evaluation..." -ForegroundColor Cyan
        & ".\venv\Scripts\python.exe" -m src.evaluation.eval_pipeline --benchmark data/eval/eval_dataset.json
    }
    "test" {
        Write-Host "--> Running pytest test suite..." -ForegroundColor Cyan
        & ".\venv\Scripts\python.exe" -m pytest -v
    }
    default {
        Write-Host "==============================================================================" -ForegroundColor Yellow
        Write-Host "                    ASK_MED POWERSHELL COMMANDS                               " -ForegroundColor Yellow
        Write-Host "==============================================================================" -ForegroundColor Yellow
        Write-Host "  .\scripts\manage.ps1 install       : Install dependencies into venv"
        Write-Host "  .\scripts\manage.ps1 docker-build  : Build Backend and UI Docker images"
        Write-Host "  .\scripts\manage.ps1 docker-up     : Start all 3 containers in background"
        Write-Host "  .\scripts\manage.ps1 docker-down   : Stop all containers"
        Write-Host "  .\scripts\manage.ps1 docker-logs   : Stream live logs"
        Write-Host "  .\scripts\manage.ps1 docker-gpu    : Launch with NVIDIA GPU acceleration"
        Write-Host "  .\scripts\manage.ps1 api           : Run FastAPI server locally (:8000)"
        Write-Host "  .\scripts\manage.ps1 ui            : Run Streamlit UI locally (:8501)"
        Write-Host "  .\scripts\manage.ps1 ingest        : Run PubMed ingestion"
        Write-Host "  .\scripts\manage.ps1 eval          : Run RAGAs benchmark evaluation"
        Write-Host "  .\scripts\manage.ps1 test          : Run full pytest suite"
        Write-Host "==============================================================================" -ForegroundColor Yellow
    }
}
