# Advanced Data Analyst Agent (Deploy-ready)

This repo contains a production-ready, deployable Streamlit app for data analysis:
- Auth-protected UI
- AutoML, visualization, forecasting (Prophet/ARIMA fallback)
- Natural-language bridge (OpenAI)
- Dockerfile + docker-compose for local testing
- GitHub Actions workflow for Render deploy

## Quick start (local)
1. Copy `.env.example` to `.env` and fill keys.
2. Build & run with docker-compose:
   ```bash
   docker compose up --build
   ```
3. Open http://localhost:8501 and enter the password from `.env`.

## Deploying
Recommended: Render (connect GitHub repo). Use the provided GitHub Actions workflow or Render's GitHub integration.
