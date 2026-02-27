import os
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from .generate_metadata import ModelMetadata

# For demonstration of loading .onnx and .pt / .ts models:
try:
    import onnxruntime as ort
except ImportError:
    ort = None  # If onnxruntime isn't installed, set it to None

try:
    import torch
except ImportError:
    torch = None  # If torch isn't installed, set it to None


class CellmapModel:
    """
    Represents a single model directory.
    Lazily loads:
      - metadata.json --> pydantic ModelMetadata
      - model.onnx    --> ONNX model session (if onnxruntime is available)
      - model.pt      --> PyTorch model (if torch is available)
      - model.ts      --> TorchScript model (if torch is available)
      - README.md      --> str
    """

    def __init__(self, folder_path: str):
        self.folder_path = folder_path

        # Internal cache for lazy properties
        self._metadata: Optional[ModelMetadata] = None
        self._readme_content: Optional[str] = None

        self._onnx_model = None
        self._pt_model = None
        self._ts_model = None
        self._exported_model = None

    @property
    def metadata(self) -> ModelMetadata:
        """Lazy load the metadata.json file and parse it into a ModelMetadata object."""
        if self._metadata is None:
            metadata_file = os.path.join(self.folder_path, "metadata.json")
            metadata_file = os.path.normpath(metadata_file)
            with open(metadata_file, "r") as f:
                data = json.load(f)
            self._metadata = ModelMetadata(**data)
        return self._metadata

    @property
    def onnx_model(self):
        """
        If 'model.onnx' exists, lazily load it as an ONNX Runtime InferenceSession.
        Use GPU if available (requires onnxruntime-gpu installed), otherwise CPU.
        Returns None if the file doesn't exist or onnxruntime isn't installed.
        """
        if self._onnx_model is None:
            model_path = os.path.join(self.folder_path, "model.onnx")
            model_path = os.path.normpath(model_path)
            if ort is None:
                # onnxruntime is not installed
                return None

            if os.path.exists(model_path):
                # Check available execution providers
                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in available_providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]

                self._onnx_model = ort.InferenceSession(model_path, providers=providers)
            else:
                self._onnx_model = None

        return self._onnx_model

    @property
    def pytorch_model(self):
        """
        If 'model.pt' exists, lazily load it using torch.load().
        Returns None if the file doesn't exist or PyTorch isn't installed.

        NOTE: Adjust this for how your .pt was saved (entire model vs state_dict).
        """
        if self._pt_model is None:
            if torch is None:
                # PyTorch is not installed
                return None
            pt_path = os.path.join(self.folder_path, "model.pt")
            pt_path = os.path.normpath(pt_path)
            if os.path.exists(pt_path):
                # Load the entire model object.
                # If your file only has the state_dict, you'll need to do something like:
                #   model = MyModelClass(...)  # define your model arch
                #   model.load_state_dict(torch.load(pt_path))
                #   self._pt_model = model
                # Instead of just torch.load().
                self._pt_model = torch.load(pt_path, weights_only=True)
            else:
                self._pt_model = None
        return self._pt_model

    @property
    def ts_model(self):
        """
        If 'model.ts' exists, lazily load it using torch.jit.load().
        Returns None if the file doesn't exist or PyTorch isn't installed.
        """
        if self._ts_model is None:
            if torch is None:
                # PyTorch is not installed
                return None
            ts_path = os.path.join(self.folder_path, "model.ts")
            ts_path = os.path.normpath(ts_path)
            if os.path.exists(ts_path):
                self._ts_model = torch.jit.load(ts_path)
            else:
                self._ts_model = None
        return self._ts_model

    @property
    def exported_model(self):
        """
        If 'model.pt2' exists, lazily load the torch.export ExportedProgram.
        Returns None if the file doesn't exist or PyTorch isn't installed.
        """
        if self._exported_model is None:
            if torch is None:
                return None
            ep_path = os.path.join(self.folder_path, "model.pt2")
            ep_path = os.path.normpath(ep_path)
            if os.path.exists(ep_path):
                self._exported_model = torch.export.load(ep_path)
            else:
                self._exported_model = None
        return self._exported_model

    def train(self):
        """
        Load a trainable nn.Module for finetuning.
        Tries torch.export (model.pt2) with unflatten first, falls back to TorchScript.
        Returns the model in train mode, or None if neither format is available.
        """
        if torch is None:
            return None

        # Try torch.export version first (unflatten to recover nn.Module hierarchy)
        if self.exported_model is not None:
            try:
                model = torch.export.unflatten(self.exported_model)
                model.train()
                print(f"Loaded trainable model via torch.export + unflatten")
                return model
            except Exception as e:
                print(f"Failed to unflatten torch.export model: {e}, falling back to TorchScript")

        # Fall back to TorchScript
        if self.ts_model is not None:
            try:
                self.ts_model.train()
                print(f"Loaded trainable model via TorchScript")
                return self.ts_model
            except Exception as e:
                print(f"Failed to set TorchScript model to train mode: {e}")

        print("No trainable model format found (model.pt2 or model.ts)")
        return None

    @property
    def readme(self) -> Optional[str]:
        """
        Lazy load the README.md content if it exists, else None.
        """
        if self._readme_content is None:
            readme_file = os.path.join(self.folder_path, "README.md")
            readme_file = os.path.normpath(readme_file)
            if os.path.exists(readme_file):
                with open(readme_file, "r", encoding="utf-8") as f:
                    self._readme_content = f.read()
            else:
                self._readme_content = None
        return self._readme_content
