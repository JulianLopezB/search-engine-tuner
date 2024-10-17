import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from .api.routes.search import router as search_router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import ssl

# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert_chain('cert.pem', keyfile='key.pem')

load_dotenv()

logger.info("Starting the application")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://contactnova-poc.netlify.app", "http://localhost:3000", '*'], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware added")

app.include_router(search_router)
logger.info("Search router included")

if __name__ == "__main__":
    logger.info(f"Running the app with SSL on host: 0.0.0.0 and port: 8000")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # ssl=ssl_context
    )