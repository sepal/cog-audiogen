# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
import tempfile

# # We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
# MODEL_PATH = "/src/models/"
# os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
# os.environ["TORCH_HOME"] = MODEL_PATH

from tempfile import TemporaryDirectory
from pathlib import Path
from cog import BasePredictor, Input, Path
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import hashlib

import torch


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model = AudioGen.get_pretrained(
            'facebook/audiogen-medium',
            device=self.__device
        )

    def set_generation_params(
            self,
            duration: int,
            top_k: int,
            top_p: float,
            temperature: float,
            classifier_free_guidance: int):
        self.__model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )

    def predict(
        self,
        prompt: str = Input(description="Prompt that describes the sound"),
        duration: float = Input(
            description="Max duration of the sound", ge=1, le=10, default=3
        ),
        top_k: int = Input(
            description="Reduces sampling to the k most likely tokens.", default=250
        ),
        top_p: float = Input(
            description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
            default=0.0,
        ),
        temperature: float = Input(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
            default=1.0,
        ),
        classifier_free_guidance: int = Input(
            description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
            default=3,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if prompt is None:
            raise ValueError("Must provide either prompt or input_audio")

        self.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            classifier_free_guidance=classifier_free_guidance,
        )
        wav = self.__model.generate([prompt])

        name = "out"
        path = f"{name}.{output_format}"

        audio_write(
            name,
            wav[0].cpu(),
            self.__model.sample_rate,
            format=output_format,
            strategy="loudness",
            loudness_compressor=True
        )


        return Path(path)
