FROM python:3.12-slim

WORKDIR /work

# System deps required by matplotlib + numpy/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY run_all.sh pytest.ini README.md CITATION.cff ./

ENV PYTHONPATH=/work

ENTRYPOINT ["bash", "run_all.sh"]
CMD ["phase1"]
