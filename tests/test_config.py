from __future__ import annotations

from pathlib import Path

import pytest

from memory_weave.config import ConfigError, load_config


def test_load_config_uses_lld_defaults() -> None:
    config = load_config()

    assert config.store.path == "./memory.sqlite"
    assert config.embedding.model == "BAAI/bge-m3"
    assert config.embedding.dims == 1024
    assert config.reranker.enabled is False
    assert config.retrieval.gate.dense_floor == 0.45
    assert config.policy.source_rank.user_statement == 4
    assert config.flags() == {
        "embedding_model": "BAAI/bge-m3",
        "embedding_version": "1",
        "rewrite_enabled": False,
        "reranker_enabled": False,
        "gate": {"dense_floor": 0.45, "lexical_min_term_fraction": 0.5},
        "reranker_floor": None,
    }


def test_load_config_applies_nested_yaml_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "memory.yaml"
    config_path.write_text(
        """\
store:
  path: /tmp/test-memory.sqlite
embedding:
  dims: 768
retrieval:
  rewrite:
    enabled: true
  gate:
    dense_floor: 0.52
reranker:
  enabled: true
  floor: 0.31
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.store.path == "/tmp/test-memory.sqlite"
    assert config.embedding.dims == 768
    assert config.retrieval.rewrite.enabled is True
    assert config.retrieval.gate.dense_floor == 0.52
    assert config.reranker.enabled is True
    assert config.reranker.floor == 0.31


def test_load_config_rejects_enabled_reranker_without_floor(tmp_path: Path) -> None:
    config_path = tmp_path / "memory.yaml"
    config_path.write_text("reranker:\n  enabled: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="reranker.floor"):
        load_config(config_path)


def test_load_config_rejects_unknown_key(tmp_path: Path) -> None:
    config_path = tmp_path / "memory.yaml"
    config_path.write_text("retrieval:\n  wrong_key: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="retrieval"):
        load_config(config_path)


def test_load_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    config_path = tmp_path / "memory.yaml"
    config_path.write_text("not_a_memory_weave_section: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="root"):
        load_config(config_path)
