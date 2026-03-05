FROM python:3.13-slim

RUN pip install --no-cache-dir openbb[all] openbb-mcp-server

EXPOSE 8001

CMD ["openbb-mcp", "--host", "0.0.0.0", "--port", "8001"]
