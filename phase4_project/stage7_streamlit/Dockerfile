FROM python:3.12-slim

WORKDIR /app

# install deps first (layer caching — deps change less than code)
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# docs for the RAG (if they're needed in the image)
COPY docs/ ./docs/

# the server listens here; 0.0.0.0 binding is set in mcp_server.py
EXPOSE 8000

# run as non-root (security)
RUN useradd -m -u 10001 appuser
USER appuser

# start the HTTP MCP server
CMD ["python", "-m", "canopy_agent.mcp_server"]
