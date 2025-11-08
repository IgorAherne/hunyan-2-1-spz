# api_spz/core/files_manage.py
import os
from pathlib import Path
from api_spz.core.state_manage import state
import logging

logger = logging.getLogger("hunyuan3d_api")

class FileManager:
    def __init__(self):
        self.base_path = state.temp_dir  # points to "temp"

        # We store everything for the active generation in "temp/current_generation"
        self.generation_path = self.base_path / "current_generation"
        self.generation_path.mkdir(parents=True, exist_ok=True)

    def get_temp_path(self, filename: str) -> Path:
        """Returns the path for a single file (e.g. 'input.png', 'model.glb')."""
        return self.generation_path / filename

    def clear_current_generation_folder(self):
        """Deletes all files in the current generation directory."""
        logger.info("Clearing current generation folder...")
        try:
            for file_name in os.listdir(self.generation_path):
                file_path = self.get_temp_path(file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            logger.error(f"Error clearing generation folder: {e}")

    def cleanup_intermediate_files(self, keep_model: bool = False):
        """Clean up intermediate files, optionally keeping the final model."""
        logger.info(f"Cleaning up intermediate files. Keep model: {keep_model}")
        try:
            for file_name in os.listdir(self.generation_path):
                # Always delete intermediate obj files, jpgs, etc.
                if file_name.endswith((".obj", ".jpg", ".png")) and "model." not in file_name:
                    os.remove(self.get_temp_path(file_name))
                # Delete final model only if not requested to keep
                elif not keep_model and (file_name.startswith("model.")):
                    os.remove(self.get_temp_path(file_name))
        except Exception as e:
            logger.error(f"Error during intermediate file cleanup: {e}")

file_manager = FileManager()