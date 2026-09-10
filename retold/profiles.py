"""Measured local-model profiles for common Retold deployments."""

from __future__ import annotations

from dataclasses import replace

from retold.config import DenseFloorConfig, EmbeddingConfig, RetoldConfig


def lite_config() -> RetoldConfig:
    """Use the calibrated MiniLM retrieval profile with Retold's standard local evidence model."""

    config = RetoldConfig(
        embedding=EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            version="1",
            dims=384,
        )
    )
    floors = DenseFloorConfig(semantic=0.32, episodic=0.46, procedural=0.30, session_summary=0.46)
    return replace(config, retrieval=replace(config.retrieval, gate=replace(config.retrieval.gate, dense_floor=floors)))
