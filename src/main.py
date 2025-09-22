# basicConfig – đủ dùng cho script / demo
import logging
from fastapi import FastAPI
import uvicorn
from api.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("app")
logger.info("App started")

app = FastAPI(
    title="RAG API",
    description="Retrieval Augmented Generation API",
    version="1.0.0"
)

app.include_router(router)

logger.info("Starting FastAPI server")
uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)