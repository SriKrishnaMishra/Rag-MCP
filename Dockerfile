FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY alembic.ini ./alembic.ini
COPY app ./app
COPY rag_core ./rag_core
COPY mcp_servers ./mcp_servers
COPY workers ./workers
RUN pip install --no-cache-dir .
RUN addgroup --system --gid 10001 app && adduser --system --uid 10001 --ingroup app --no-create-home app
ENV PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
