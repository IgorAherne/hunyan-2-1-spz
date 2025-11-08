# hy3dpaint\textureGenPipeline.py

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import gc
import torch
import copy
import trimesh
import numpy as np
from PIL import Image
from typing import List
from DifferentiableRenderer.MeshRender import MeshRender
from utils.simplify_mesh_utils import remesh_mesh
from utils.multiview_utils import multiviewDiffusionNet
from utils.pipeline_utils import ViewProcessor
from utils.image_super_utils import imageSuperNet
from utils.uvwrap_utils import mesh_uv_wrap
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)


class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view=8, resolution=768, view_chunk_size=8):
        self.device = "cuda"

        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.max_selected_view_num = max_num_view
        self.view_chunk_size = view_chunk_size if view_chunk_size > 0 else max_num_view
        self.resolution = resolution
        self.bake_exp = 4
        self.merge_method = "fast"

        # view selection
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)

            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)


class Hunyuan3DPaintPipeline:

    def __init__(self, config=None) -> None:
        self.config = config if config is not None else Hunyuan3DPaintConfig()
        self.models = {} # Initialize empty models dict
        self.is_compiled = False # Default state
        self.sentinel_path = None # Always initialize the attribute

        # Check for the on-disk cache sentinel file to avoid redundant warm-ups
        if 'TORCHINDUCTOR_CACHE_DIR' in os.environ:
            cache_dir = os.environ['TORCHINDUCTOR_CACHE_DIR']
            self.sentinel_path = os.path.join(cache_dir, "_warmup_complete.sentinel")
            if os.path.exists(self.sentinel_path):
                print("[INFO] On-disk compilation cache found. Skipping warm-up.")
                self.is_compiled = True
        
        self.stats_logs = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
        )
        self.view_processor = ViewProcessor(self.config, self.render)
        # Do not load models automatically in __init__
        # self.load_models()

    def load_model(self, model_name):
        """Load a specific model on demand."""
        if model_name not in self.models:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if model_name == "super_model":
                super_model_net = imageSuperNet(self.config)
                # Apply torch.compile to the underlying RRDBNet model within the RealESRGANer
                super_model_net.upsampler.model = torch.compile(super_model_net.upsampler.model)
                self.models[model_name] = super_model_net
            elif model_name == "multiview_model":
                # Load multiview model (handles Bfloat16 and CPU offloading internally)
                self.models[model_name] = multiviewDiffusionNet(self.config)
            print(f"{model_name} Loaded.")

    def offload_model(self, model_name):
        """Offload a specific model from the GPU."""
        if model_name in self.models:
            # Use the specific free_memory method if available
            if hasattr(self.models[model_name], 'free_memory'):
                self.models[model_name].free_memory()
            elif hasattr(self.models[model_name], 'to'):
                 self.models[model_name].to("cpu")
            
            del self.models[model_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"{model_name} Offloaded.")

    def free_memory(self):
        """Frees up all memory."""
        self.offload_model("super_model")
        self.offload_model("multiview_model")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Hunyuan3DPaintPipeline memory freed.")


    def _jit_compile_warmup(self):
        """
        Performs a one-time, low-overhead run to trigger torch.compile JIT compilation.
        This avoids a long pause on the first real generation. Caches the result on disk.
        """
        if self.is_compiled:
            return

        print("[INFO] Performing one-time JIT compilation warm-up. This may take a moment...")
        try:
            # Create minimal dummy inputs for the fastest possible run
            warmup_size = 64
            dummy_style_image = [Image.new('RGB', (warmup_size, warmup_size), 'white')]
            dummy_conditions = [Image.new('RGB', (warmup_size, warmup_size), 'white')] * 2  # Normal + Position

            # Execute a single pass with 1 view and 1 step to trigger compilation
            _ = self.models["multiview_model"](
                dummy_style_image,
                dummy_conditions,
                prompt="warmup",
                custom_view_size=warmup_size,
                resize_input=True,
                num_inference_steps=1
            )
            # If the warm-up was successful, create the sentinel file (to prevent re-compiles if server restarts)
            with open(self.sentinel_path, 'w') as f:
                pass  # Create an empty file to mark success
            print(f"[INFO] Sentinel file created at {self.sentinel_path}")

            print("[INFO] JIT warm-up complete. Main generation will now proceed.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARNING] JIT warm-up failed: {e}. Proceeding without compilation.")
        finally:
            self.is_compiled = True  # Mark as compiled for this session to prevent re-running
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    @torch.no_grad()
    def __call__(self, mesh_path=None, image_path=None, output_mesh_path=None, use_remesh=True, save_glb=True):
        """Generate texture for 3D mesh using multiview diffusion"""

         # --- One-Time JIT Compilation Warm-up ---
        self.load_model("multiview_model") # Ensure model is loaded before warm-up
        self._jit_compile_warmup()
        # --- End of Warm-up ---

        # Ensure image_prompt is a list
        if isinstance(image_path, str):
            image_prompt = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image_prompt = image_path
        
        # Handle the case where image_path is a list
        if isinstance(image_path, list):
             image_prompt = image_path
        elif not isinstance(image_prompt, List):
            image_prompt = [image_prompt]
        else:
            image_prompt = image_path

        # Process mesh
        path = os.path.dirname(mesh_path)
        if use_remesh:
            processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
            remesh_mesh(mesh_path, processed_mesh_path)
        else:
            processed_mesh_path = mesh_path

        # Output path
        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, f"textured_mesh.obj")

        # Load mesh
        mesh = trimesh.load(processed_mesh_path)
        mesh = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh=mesh)

        ########### View Selection #########
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )

        print(f"Requested a maximum of {self.config.max_selected_view_num} views. bake_view_selection returned {len(selected_camera_elevs)} views.")

        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)

        ##########  Style  ###########
        image_caption = "high quality"
        image_style = []
        for image in image_prompt:
            image = image.resize((512, 512))
            if image.mode == "RGBA":
                white_bg = Image.new("RGB", image.size, (255, 255, 255))
                white_bg.paste(image, mask=image.getchannel("A"))
                image = white_bg
            image_style.append(image)
        image_style = [image.convert("RGB") for image in image_style]

        ###########  Multiview  ##########
        # Load multiview model sequentially
        self.load_model("multiview_model")
        
        all_multiviews_pbr = {"albedo": [], "mr": []}
        num_views = len(selected_camera_elevs)
        chunk_size = self.config.view_chunk_size
        persistent_cache = {} # Create a cache that will persist across chunks

        # not processing all the views at once
        for i in tqdm(range(0, num_views, chunk_size), desc="Processing views in chunks", position=0, leave=True):
            chunk_end = min(i + chunk_size, num_views)
        
            # Slice the condition maps for the current chunk
            chunk_normal_maps = normal_maps[i:chunk_end]
            chunk_position_maps = position_maps[i:chunk_end]

            print(f"Processing chunk {i//chunk_size + 1}/{(num_views + chunk_size - 1)//chunk_size} with {len(chunk_normal_maps)} views...")
            
            chunk_multiviews_pbr = self.models["multiview_model"](
                image_style,
                chunk_normal_maps + chunk_position_maps,
                prompt=image_caption,
                custom_view_size=self.config.resolution,
                resize_input=True,
                cache=persistent_cache
            )
            
            # Append results from the chunk
            all_multiviews_pbr["albedo"].extend(chunk_multiviews_pbr.get("albedo", []))
            if "mr" in chunk_multiviews_pbr:
                all_multiviews_pbr["mr"].extend(chunk_multiviews_pbr.get("mr", []))

            # Aggressive memory clearing between chunks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        multiviews_pbr = all_multiviews_pbr
        
        # Offload multiview model
        self.offload_model("multiview_model")

        ###########  Enhance  ##########
        enhance_images = {}
        enhance_images["albedo"] = copy.deepcopy(multiviews_pbr["albedo"])
        # Check if 'mr' exists before processing
        if "mr" in multiviews_pbr:
            enhance_images["mr"] = copy.deepcopy(multiviews_pbr["mr"])

        # Load super-resolution model sequentially
        self.load_model("super_model")
        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = self.models["super_model"](enhance_images["albedo"][i])
            if "mr" in enhance_images:
                enhance_images["mr"][i] = self.models["super_model"](enhance_images["mr"][i])
        
        # Offload super-resolution model
        self.offload_model("super_model")

        ###########  Bake  ##########
        # Ensure correct loop iteration count
        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                (self.config.render_size, self.config.render_size)
            )
            if "mr" in enhance_images:
                enhance_images["mr"][i] = enhance_images["mr"][i].resize((self.config.render_size, self.config.render_size))
        
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        
        if "mr" in enhance_images:
            texture_mr, mask_mr = self.view_processor.bake_from_multiview(
                enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )
            mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        ##########  inpaint  ###########
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture, force_set=True)
        if "mr" in enhance_images:
            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            self.render.set_texture_mr(texture_mr)

        self.render.save_mesh(output_mesh_path, downsample=True)

        if save_glb:
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))
            output_glb_path = output_mesh_path.replace(".obj", ".glb")

        return output_mesh_path
