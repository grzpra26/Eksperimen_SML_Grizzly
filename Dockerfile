FROM python:3.12.7-slim

WORKDIR /app

COPY MLProject/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY MLProject/ /app/

EXPOSE 8000

CMD ["python", "7.Inference.py"]
