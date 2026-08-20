"""
AWS Lambda entry point for ContractIQ API.

Usage:
  Set Lambda handler to: lambda_handler.handler

The Mangum adapter translates API Gateway events into ASGI
requests that FastAPI can process.
"""

from mangum import Mangum
from api.main import app

# Mangum wraps the FastAPI ASGI app for Lambda
# lifespan="off" skips the startup/shutdown events
# (model loading is handled lazily on first request instead)
handler = Mangum(app, lifespan="off")
