FROM python:3.15-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py llm.py pdf_text.py ingest.py query.py main.py ./

CMD ["python", "main.py"]
