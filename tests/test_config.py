from __future__ import annotations

from pathlib import Path

import pytest

from memory_weave.config import ConfigError, load_config


def test_load_config_uses_lld_defaults() -> None:
    config = load_config()

    assert config.store.path == "./memory.sqlite"
    assert config.embedding.model == "BAAI/bge-m3"
    assert config.embedding.dims == 1024
    assert config.embedding.query_cache_entries == 4096
    assert config.reranker.enabled is False
    assert config.retrieval.gate.dense_floor.semantic == 0.45
    assert config.retrieval.trigger.mode == "tool_only"
    assert config.policy.source_rank.user_statement == 4
    assert config.flags() == {
        "embedding_model": "BAAI/bge-m3",
        "embedding_version": "1",
        "embedding_query_cache_entries": 4096,
        "rewrite_enabled": False,
        "reranker_enabled": False,
        "trigger_mode": "tool_only",
        "gate": {
            "dense_floor": {"semantic": 0.45, "episodic": 0.40, "procedural": 0.45},
            "lexical_min_term_fraction": 0.5,
            "lexical_min_matched_terms": 2,
            "relative_floor": 0.5,
        },
        "reranker_floor": None,
    }


def test_load_config_accepts_every_trigger_mode_and_rejects_unknown_ones(tmp_path: Path) -> None:
    for mode in ("tool_only", "auto", "hybrid"):
        config_path = tmp_path / f"{mode}.yaml"
        config_path.write_text(f"retrieval:\n  trigger:\n    mode: {mode}\n    auto_k: 3\n", encoding="utf-8")
        config = load_config(config_path)
        assert config.retrieval.trigger.mode == mode
        assert config.retrieval.trigger.auto_k == 3
        assert config.flags()["trigger_mode"] == mode

    config_path = tmp_path / "bad.yaml"
    config_path.write_text("retrieval:\n  trigger:\n    mode: ambient\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="trigger.mode"):
        load_config(config_path)

    config_path.write_text("retrieval:\n  gate:\n    relative_floor: 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="relative_floor"):
        load_config(config_path)

    config_path.write_text("embedding:\n  query_cache_entries: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="query_cache_entries"):
        load_config(config_path)


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
    dense_floor:
      semantic: 0.52
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
    assert config.retrieval.gate.dense_floor.semantic == 0.52
    assert config.retrieval.gate.dense_floor.episodic == 0.40
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


def test_load_config_coerces_numeric_yaml_scalars_from_annotations(tmp_path: Path) -> None:
    config_path = tmp_path / "memory.yaml"
    config_path.write_text(
        "embedding:\n  dims: '768'\n"
        "retrieval:\n  per_generator_k: '40'\n  gate:\n    dense_floor:\n      semantic: '0.52'\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.embedding.dims == 768
    assert isinstance(config.embedding.dims, int)
    assert config.retrieval.per_generator_k == 40
    assert isinstance(config.retrieval.per_generator_k, int)
    assert config.retrieval.gate.dense_floor.semantic == 0.52
    assert isinstance(config.retrieval.gate.dense_floor.semantic, float)


def test_load_config_rejects_invalid_numeric_yaml_scalars_with_config_error(tmp_path: Path) -> None:
    config_path = tmp_path / "memory.yaml"
    config_path.write_text("retrieval:\n  per_generator_k: 40.5\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="retrieval.per_generator_k"):
        load_config(config_path)
