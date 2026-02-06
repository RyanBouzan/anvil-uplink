FROM python:3.8

# Put files in a predictable place
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt ./requirements.txt

# Create your folder (and open perms like you had)
RUN mkdir -p /dockerfolder && chmod -R 777 /dockerfolder

# Copy your code (optional, but keeping what you had)
COPY connect4_backend.py ./connect4_backend.py

ENV PYTHONUNBUFFERED=1

# Create an entrypoint that tries to install deps but DOES NOT fail the container
RUN printf '%s\n' \
  '#!/bin/sh' \
  'set -e' \
  'echo "Attempting pip install (will not crash if it fails)..."' \
  'pip install --no-cache-dir -r /app/requirements.txt || echo "pip install failed (likely no internet). Continuing...";' \
  'echo "Container is staying alive. Exec in with: docker exec -it <name> bash"' \
  'sleep infinity' \
  > /entrypoint.sh && chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
