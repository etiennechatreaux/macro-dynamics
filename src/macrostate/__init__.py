"""Macro regime detection preprocessing pipeline."""
from macrostate.config.settings import PreprocessConfig
from macrostate.pipelines.preprocess import build_preprocessing_pipeline

__version__ = "0.1.6"
__all__ = ["PreprocessConfig", "build_preprocessing_pipeline"]

