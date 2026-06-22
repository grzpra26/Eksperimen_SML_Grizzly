FROM python:3.12.7-slim

WORKDIR /app

COPY MLProject/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY MLProject/ /app/

EXPOSE 8000

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
