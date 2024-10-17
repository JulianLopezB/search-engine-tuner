import logging
from dotenv import load_dotenv
import ssl

    
load_dotenv()

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def start():
    """Launched with `poetry run start` at root level"""
    import uvicorn
    import multiprocessing
    logger.info("Starting the application in debug mode")
    # Disable multiprocessing
    multiprocessing.freeze_support()
    uvicorn.run("search_engine.main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug",  workers=1)

if __name__ == "__main__"
    start()