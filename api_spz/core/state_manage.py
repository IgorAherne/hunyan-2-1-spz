# api_spz/core/state_manage.py

import logging
import torch
import trimesh
from pathlib import Path

from hy3dshape import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FloaterRemover,
    DegenerateFaceRemover,
    FaceReducer,
)
from hy3dshape.rembg import BackgroundRemover

logger = logging.getLogger("hunyuan3d_api")

class HunyuanState:
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.pipeline_config = {}
        self.rembg = None
        self.floater_remover = None
        self.degenerate_face_remover = None
        self.face_reducer = None
        self.device = "cpu"

    def initialize_utilities(
        self,
        model_path="tencent/Hunyuan3D-2.1",
        subfolder="hunyuan3d-dit-v2-1",
        device=None,
        enable_flashvdm=True,
        mc_algo="mc",
    ):
        """
        Initializes all necessary components for the API.
        The main pipelines are not loaded here to save memory; they are loaded on-demand.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing utilities for device: {self.device}")

        # Store pipeline configuration to be used by the worker process
        self.pipeline_config = {
            "model_path": model_path,
            "subfolder": subfolder,
            "device": self.device,
            "enable_flashvdm": enable_flashvdm,
            "mc_algo": mc_algo,
        }

        # Load lightweight utility workers
        self.rembg = BackgroundRemover()
        self.floater_remover = FloaterRemover()
        self.degenerate_face_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()

        logger.info("HunyuanState utilities initialized successfully.")

    @staticmethod
    def run_shape_generation_worker(queue, args_dict):
        """
        This function runs in a separate process to generate the shape,
        ensuring all memory is released upon completion.
        It's a self-contained version of the logic from gradio_app.py.
        """
        try:
            # Re-import everything within the new process
            import torch
            import trimesh
            from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dshape.pipelines import export_to_trimesh

            print("Worker Process: Loading shape generation pipeline...")
            shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                args_dict['model_path'],
                subfolder=args_dict['subfolder'],
                use_safetensors=True,
                device=args_dict['device'],
                dtype=torch.float16,
            )
            if args_dict['enable_flashvdm']:
                mc_algo_val = 'mc' if args_dict['device'] in ['cpu', 'mps'] else args_dict['mc_algo']
                shape_pipeline.enable_flashvdm(mc_algo=mc_algo_val)

            generator = torch.Generator(device=args_dict.get('device', 'cpu'))
            generator.manual_seed(int(args_dict['seed']))

            outputs = shape_pipeline(
                image=args_dict['image'],
                num_inference_steps=args_dict['steps'],
                guidance_scale=args_dict['guidance_scale'],
                generator=generator,
                octree_resolution=args_dict['octree_resolution'],
                num_chunks=args_dict['num_chunks'],
                output_type='mesh',
            )
            
            mesh = export_to_trimesh(outputs)[0]

            stats = {
                'number_of_faces': mesh.faces.shape[0],
                'number_of_vertices': mesh.vertices.shape[0],
            }

            print("Worker Process: Freeing shape generation pipeline memory...")
            shape_pipeline.free_memory()
            del shape_pipeline

            # Send pickleable data back to the main process
            mesh_data = (mesh.vertices, mesh.faces)
            queue.put(('success', mesh_data, stats))

        except Exception as e:
            import traceback
            traceback.print_exc()
            queue.put(('error', str(e)))
        finally:
            if 'queue' in locals():
                print("Worker Process: Closing queue.")
                queue.close()
                queue.join_thread()

# Global state instance
state = HunyuanState()