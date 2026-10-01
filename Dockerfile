FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /lab
# pinned for reproducibility (2026-10 best practice: pin base + reqs, no latest)
COPY requirements.txt /lab/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .
COPY . /lab
CMD ["bash", "-lc", "pytest -q && python experiments/exp01_stochastic.py"]
