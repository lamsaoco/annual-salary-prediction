# Use python ver 3.12.10
FROM python:3.12.10-slim-bookworm

# Install uv environment
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container to /app
WORKDIR /app

# Add the virtual environment's bin directory to the PATH so Python tools work globally
ENV PATH="/app/.venv/bin:$PATH"

# Copy the project configuration files into the container
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install dependencies exactly as locked in uv.lock, without updating them
RUN uv sync --locked

# Copy application source code and model artifact into the container
COPY "src/predict.py" ./src/predict.py
COPY "models/ml_xgboost.bin" ./models/ml_xgboost.bin

# Expose TCP port 9696 so it can be accessed from outside the container
EXPOSE 9696

ENTRYPOINT ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "9696"]