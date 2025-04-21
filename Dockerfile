FROM python:3.12

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install uv
RUN pip install uv

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

RUN uv init
RUN uv add numpy plotly prophet streamlit pandas yfinance

EXPOSE 8501

COPY main.py .
COPY src/ src/

CMD ["uv", "run", "streamlit", "run", "--server.address=0.0.0.0", "--server.port=8501", "main.py"]