# Stop on first error
$ErrorActionPreference = "Stop"

Write-Host "CLEANING UP OLD DATA..." -ForegroundColor Red
Remove-Item -Path "backend/data/processed/*.json" -ErrorAction SilentlyContinue
Remove-Item -Path "backend/data/processed/*.parquet" -ErrorAction SilentlyContinue
Remove-Item -Path "backend/data/taxonomy/*.json" -ErrorAction SilentlyContinue
Remove-Item -Path "backend/models/*.pkl" -ErrorAction SilentlyContinue
Remove-Item -Path "backend/models/*.txt" -ErrorAction SilentlyContinue
Remove-Item -Path "backend/indices/*.index" -ErrorAction SilentlyContinue

Write-Host "BUILDING GEO DATABASE..." -ForegroundColor Cyan
python scripts/build_geo_db.py

Write-Host "GENERATING PROFILES..." -ForegroundColor Cyan
python scripts/generate_snp_profiles.py

Write-Host "GENERATING TRAINING DATA (LTR)..." -ForegroundColor Cyan
python scripts/generate_ltr_pairs.py

Write-Host "TRAINING CLASSIFIER..." -ForegroundColor Cyan
python training/train_classifier.py

Write-Host "TRAINING RANKER..." -ForegroundColor Cyan
python training/train_ltr.py

Write-Host "BUILDING SEARCH INDEX..." -ForegroundColor Cyan
python scripts/build_faiss_index.py

Write-Host "BUILD COMPLETE. STARTING SERVER..." -ForegroundColor Green
uvicorn backend.app.main:app --reload --port 8000