# Matches the local dev Python version (3.9) to avoid any risk of subtle
# pickle/numpy incompatibility when unpickling the joblib model artifact
# trained locally under 3.9.
FROM python:3.9-slim

WORKDIR /app

COPY ml/model_api/requirements.txt ./ml/model_api/requirements.txt
RUN pip install --no-cache-dir -r ml/model_api/requirements.txt

COPY features/feature_store.py ./features/feature_store.py
COPY ml/model_api/main.py ./ml/model_api/main.py
COPY ml/model_api/model.joblib ./ml/model_api/model.joblib

EXPOSE 8000
CMD ["uvicorn", "ml.model_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
