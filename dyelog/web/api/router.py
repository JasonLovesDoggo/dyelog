from fastapi.routing import APIRouter

from dyelog.web.api import chat, docs, monitoring, speech

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(docs.router)
api_router.include_router(chat.router)
api_router.include_router(speech.router, tags=["speech"])
