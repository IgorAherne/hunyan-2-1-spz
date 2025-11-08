# api_spz/routes/generation.py

import logging
import time
import traceback
from typing import Optional, List
import asyncio
import io
import base64
import trimesh
from fastapi import APIRouter, File, Response, UploadFile, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from PIL import Image
from multiprocessing import Process, Queue

from api_spz.core.files_manage import file_manager
from api_spz.core.state_manage import state
from api_spz.core.models_pydantic import (
    GenerationArgForm,
    GenerationResponse,
    TaskStatus,
    StatusResponse,
)
# Use explicit path to import directly from the library's module
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

router = APIRouter()
logger = logging.getLogger("hunyuan3d_api")

generation_lock = asyncio.Lock()
current_generation = {
    "status": TaskStatus.FAILED,
    "progress": 0,
    "message": "",
}

def is_generation_in_progress() -> bool:
    return generation_lock.locked()

def reset_current_generation():
    current_generation.update({
        "status": TaskStatus.PROCESSING,
        "progress": 0,
        "message": "",
    })

def update_current_generation(status: Optional[TaskStatus] = None, progress: Optional[int] = None, message: Optional[str] = None):
    if status: current_generation["status"] = status
    if progress: current_generation["progress"] = progress
    if message: current_generation["message"] = message

def normalize_mesh_simplify_ratio(ratio: float) -> float:
    return ratio / 100.0 if ratio > 1.0 else ratio

async def _load_images_from_request(
    files: Optional[List[UploadFile]] = None,
    images_base64: Optional[List[str]] = None
) -> List[Image.Image]:
    images = []
    files = files or []
    images_base64 = images_base64 or []
    
    for b64_str in images_base64:
        try:
            if "base64," in b64_str:
                b64_str = b64_str.split("base64,")[1]
            img_bytes = base64.b64decode(b64_str)
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGBA"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    for file in files:
        try:
            content = await file.read()
            images.append(Image.open(io.BytesIO(content)).convert("RGBA"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read image file: {e}")

    if not images:
        raise HTTPException(status_code=400, detail="No valid images provided.")
    return images

async def _run_generation_task(pil_images: List[Image.Image], arg: GenerationArgForm):
    """Orchestrates the entire 3D model generation process."""
    update_current_generation(progress=5, message="Preparing for shape generation...")

    # 1. Prepare image(s) for the shape generation worker
    image_for_worker = None
    is_multiview_model = 'mv' in state.pipeline_config.get('model_path', '').lower()

    if is_multiview_model:
        logger.info("Multi-view model detected. Creating view dictionary.")
        image_for_worker = {}
        view_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        for i, img in enumerate(pil_images):
            if i < len(view_names):
                processed_img = state.rembg(img.convert('RGB'))
                image_for_worker[view_names[i]] = processed_img
    else:
        if len(pil_images) > 1:
            logger.warning("Multiple images provided for a single-view model. Using only the first image.")
        image_for_worker = state.rembg(pil_images[0].convert('RGB'))

    # 2. Run Shape Generation in a separate process
    update_current_generation(progress=10, message="Generating 3D shape (this may take a few minutes)...")
    args_dict = {
        **state.pipeline_config,
        'image': image_for_worker,
        'steps': arg.num_inference_steps,
        'guidance_scale': arg.guidance_scale,
        'seed': arg.seed,
        'octree_resolution': arg.octree_resolution,
        'num_chunks': arg.num_chunks,
    }

    queue = Queue()
    process = Process(target=state.run_shape_generation_worker, args=(queue, args_dict))
    process.start()
    result = await asyncio.to_thread(queue.get)
    process.join()

    if not result or result[0] == 'error':
        raise Exception(f"Shape generation failed: {result[1] if result else 'No result returned'}")

    _, mesh_data, _ = result
    mesh = trimesh.Trimesh(vertices=mesh_data[0], faces=mesh_data[1])

    update_current_generation(progress=50, message="Shape generated. Post-processing...")
    mesh = state.floater_remover(mesh)
    mesh = state.degenerate_face_remover(mesh)

    # 3. Texture Generation (if requested)
    if arg.apply_texture:
        update_current_generation(progress=60, message="Applying texture...")
        
        temp_obj_path = file_manager.get_temp_path("temp_for_texture.obj")
        mesh.export(str(temp_obj_path))
        
        # Configure and run the texture pipeline
        conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=768, view_chunk_size=3)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        conf.texture_size = arg.texture_size
        texture_pipeline = Hunyuan3DPaintPipeline(conf)
        
        output_textured_obj_path = file_manager.get_temp_path("textured_mesh.obj")
        
        def texture_worker():
            try:
                texture_pipeline(
                    mesh_path=str(temp_obj_path), 
                    image_path=pil_images[0], 
                    output_mesh_path=str(output_textured_obj_path), 
                    save_glb=False
                )
            finally:
                texture_pipeline.free_memory()

        await asyncio.to_thread(texture_worker)
        mesh = trimesh.load(str(output_textured_obj_path), force="mesh")
        update_current_generation(progress=85, message="Texture applied.")
    
    # 4. Mesh Simplification
    simplify_ratio = normalize_mesh_simplify_ratio(arg.mesh_simplify_ratio)
    if simplify_ratio < 1.0:
        update_current_generation(progress=90, message="Simplifying mesh...")
        target_faces = int(len(mesh.faces) * simplify_ratio)
        mesh = state.face_reducer(mesh, max_facenum=target_faces)
    
    # 5. Export final model
    update_current_generation(progress=95, message="Exporting final model...")
    model_path = file_manager.get_temp_path(f"model.{arg.output_format}")
    mesh.export(str(model_path))
    return str(model_path)

async def generation_endpoint_logic(images: List[Image.Image], arg: GenerationArgForm):
    if is_generation_in_progress():
        raise HTTPException(status_code=503, detail="Server is busy with another generation.")
    
    async with generation_lock:
        start_time = time.time()
        reset_current_generation()
        try:
            await _run_generation_task(images, arg)
            update_current_generation(status=TaskStatus.COMPLETE, progress=100, message="Generation complete")
            duration = time.time() - start_time
            logger.info(f"Generation completed in {duration:.2f} seconds")
            return GenerationResponse(
                status=TaskStatus.COMPLETE,
                progress=100,
                message="Generation complete",
                model_url="/download/model"
            )
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Generation failed: {e}\n{error_trace}")
            update_current_generation(status=TaskStatus.FAILED, message=str(e))
            raise HTTPException(status_code=500, detail=str(e))

@router.get("/ping")
async def ping():
    return {"status": "running", "message": "API is operational", "busy": is_generation_in_progress()}

@router.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(**current_generation, busy=is_generation_in_progress())

@router.post("/generate_no_preview", response_model=GenerationResponse)
async def generate_no_preview(file: UploadFile = File(...), arg: GenerationArgForm = Depends()):
    logger.info("Received single image generation request.")
    images = await _load_images_from_request(files=[file])
    return await generation_endpoint_logic(images, arg)

@router.post("/generate_multi_no_preview", response_model=GenerationResponse)
async def generate_multi_no_preview(file_list: List[UploadFile] = File(...), arg: GenerationArgForm = Depends()):
    logger.info("Received multi-view generation request.")
    images = await _load_images_from_request(files=file_list)
    return await generation_endpoint_logic(images, arg)

@router.post("/generate", response_model=GenerationResponse)
async def process_ui_generation_request(data: dict):
    """Process generation request from the StableProjectorz UI panel."""
    logger.info("Processing UI generation request.")
    try:
        arg = GenerationArgForm(
            seed=int(data.get("seed", 1234)),
            guidance_scale=float(data.get("guidance_scale", 5.0)),
            num_inference_steps=int(data.get("num_inference_steps", 20)),
            octree_resolution=int(data.get("octree_resolution", 256)),
            num_chunks=int(data.get("num_chunks", 80)),
            mesh_simplify_ratio=float(data.get("mesh_simplify", 10.0)),
            apply_texture=bool(data.get("apply_texture", True)),
            texture_size=int(data.get("texture_size", 2048)),
        )
        images = await _load_images_from_request(images_base64=data.get("single_multi_img_input", []))
        return await generation_endpoint_logic(images, arg)
    except Exception as e:
        logger.error(f"Error in UI generation request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info/supported_operations")
async def get_supported_operations():
    return ["make_meshes_and_tex"]

@router.get("/download/model")
async def download_model():
    model_path = file_manager.get_temp_path("model.glb")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")
    return FileResponse(str(model_path), media_type="model/gltf-binary", filename="model.glb")

@router.get("/download/spz-ui-layout/generation-3d-panel")
async def get_generation_panel_layout():
    try:
        # Assuming the layout file is in the same directory as this script
        file_path = "api_spz/routes/layout_generation_3d_panel.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content=content, media_type="text/plain; charset=utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Layout file not found")