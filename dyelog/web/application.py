from __future__ import annotations

import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import UJSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import Response

from dyelog.web.api.router import api_router
from dyelog.web.lifespan import lifespan_setup

APP_ROOT = Path(__file__).parent.parent


def get_app() -> FastAPI:
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    app = FastAPI(
        title="dyelog",
        version=metadata.version("dyelog"),
        lifespan=lifespan_setup,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        default_response_class=UJSONResponse,
    )

    async def catch_exceptions_middleware(
        request: Request,
        call_next: Any,
    ) -> Exception | Response:
        try:
            return await call_next(request)
        except Exception as e:
            print(traceback.format_exc())
            raise e

    app.middleware("http")(catch_exceptions_middleware)
    # Main router for the API.
    app.include_router(router=api_router, prefix="/api")
    # Adds static directory.
    # This directory is used to access swagger files.
    app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")

    return app
