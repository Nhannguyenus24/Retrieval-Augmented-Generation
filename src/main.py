# basicConfig – đủ dùng cho script / demo
import logging
from fastapi import FastAPI
import uvicorn
from api.router import router
import dotenv 

dotenv.load_dotenv()

logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="RAG API",
    description="Retrieval Augmented Generation API",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    logger.info("Starting FastAPI server (no reload)")
    uvicorn.run(app, host="0.0.0.0", port=1234, reload=False)