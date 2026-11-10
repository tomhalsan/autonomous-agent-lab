FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app/contracts:/app/policy:/app/infrastructure
WORKDIR /app
COPY services/research_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/contracts/src /app/contracts
COPY packages/policy/src /app/policy
COPY packages/infrastructure/src /app/infrastructure
COPY services/research_agent/src /app/src
CMD ["python", "-m", "research_agent.main"]

