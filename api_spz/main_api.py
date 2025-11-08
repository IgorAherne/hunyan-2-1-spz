# main_api.py

import logging
import os
import sys

# Add the parent directory to sys.path to allow imports from api_spz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Apply torchvision compatibility fix before other imports
try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix.")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

import platform
import torch
import multiprocessing as mp
from contextlib import asynccontextmanager

import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_spz.routes.generation import router as generation_router
from api_spz.core.state_manage import state

# Setup Logging
os.makedirs("temp", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("temp", "api.log"))
    ]
)
logger = logging.getLogger("hunyuan3d_api")

# System Info
print(
    f"\n[System Info] Python: {platform.python_version():<8} | "
    f"PyTorch: {torch.__version__:<8} | "
    f"CUDA: {'not available' if not torch.cuda.is_available() else torch.version.cuda}\n"
)

# Argument Parsing
parser = argparse.ArgumentParser(description="Run Hunyuan3D-StableProjectorz API server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
parser.add_argument("--port", type=int, default=7960, help="Port to bind the server to")
parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2.1', help="Hunyuan3D model path")
parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-1', help="Model subfolder")
parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
parser.add_argument("--enable_flashvdm", action='store_true', default=True, help="Enable FlashVDM acceleration")
args, _ = parser.parse_known_args()

# Startup Info
print("\n" + "="*50)
print("Hunyuan3D-StableProjectorz API Server is starting...")
print("If it's the first time, models will be downloaded. This may take a while.")
print("="*50 + "\n")

# FastAPI Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models and resources
    state.initialize_utilities(
        model_path=args.model_path,
        subfolder=args.subfolder,
        device=args.device,
        enable_flashvdm=args.enable_flashvdm,
    )
    
    print("\n" + "="*50)
    print("Hunyuan3D-StableProjectorz API Server v2.1.0")
    print(f"Server is active and listening on http://{args.host}:{args.port}")
    print("In StableProjectorz, go to 3D mode, click the connection button, and enter the address.")
    print("="*50 + "\n")
    
    yield
    
    # Shutdown: Clean up (if needed)
    logger.info("Server shutting down.")

# FastAPI App Initialization
app = FastAPI(
    title="Hunyuan3D-StableProjectorz API",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generation_router)

@app.get("/")
async def root():
    return {"message": "Hunyuan3D-StableProjectorz API is running."}

if __name__ == "__main__":
    # CRITICAL: Set start method to 'spawn' for CUDA safety.
    # This must be done in the main execution block.
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method was already set.")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")