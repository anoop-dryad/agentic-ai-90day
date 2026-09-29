# ---- config (values come from environment / .envrc, never hardcoded) ----
IMAGE       ?= canopy-mcp-server
PORT        ?= 8000
BACKEND_URL ?= http://host.docker.internal:8080/v1

# ---- local dev ----
install:
	pip install -e ".[dev]"

run-server:
	python -m canopy_agent.mcp_server

run-client:
	python -m canopy_agent.mcp_client_agent

test:
	pytest -m "not llm" -v

lint:
	ruff check src/ tests/

# ---- docker ----
docker-build:
	docker build -t $(IMAGE) .

docker-run:
	@test -n "$(CANOPY_API_KEY)" || { echo "CANOPY_API_KEY not set — run 'direnv allow'"; exit 1; }
	@test -n "$(GEMINI_API_KEY)" || { echo "GEMINI_API_KEY not set — run 'direnv allow'"; exit 1; }
	docker run --rm -p $(PORT):8000 \
		-e CANOPY_API_KEY="$(CANOPY_API_KEY)" \
		-e GEMINI_API_KEY="$(GEMINI_API_KEY)" \
		-e BACKEND_BASE_URL="$(BACKEND_URL)" \
		-e DEVICE_PATH="devices" \
		$(IMAGE)
