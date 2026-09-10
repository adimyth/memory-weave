"""Calibrate one Sentence Transformers embedding model on Retold's labelled 1K fixture.

The script rebuilds the fixture in a temporary directory, so it never changes the committed BGE-M3 fixture.
It reports retrieval recall and the dense-floor F1 sweep used to decide whether a model profile is supportable.

Run the lightweight candidate with:

    HF_HUB_OFFLINE=1 uv run --extra local-models python benchmarks/calibrate_embedding.py \
        --model sentence-transformers/all-MiniLM-L6-v2 --dims 384
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from retold.config import DenseFloorConfig, EmbeddingConfig, RetoldConfig
from retold.index.embedder import BgeM3Embedder
from retold.index.vector import VectorIndex
from retold.models import Principal, SearchRequest
from retold.policy import readable_scopes
from retold.runtime import build_runtime
from retold.store import Store
from tests.integration.fixture_1k import NOW, build_fixture

QUERIES = Path(__file__).resolve().parents[1] / "tests" / "integration" / "labelled_queries.yaml"


def _labelled() -> dict[str, list[dict[str, Any]]]:
    loaded = yaml.safe_load(QUERIES.read_text(encoding="utf-8"))
    if len(loaded["relevant"]) != 50 or len(loaded["irrelevant"]) != 50:
        raise ValueError("The embedding calibration requires 50 relevant and 50 irrelevant queries.")
    return loaded  # type: ignore[no-any-return]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    embedding = EmbeddingConfig(model=args.model, version="calibration-1", dims=args.dims, device=args.device)
    config = RetoldConfig(embedding=embedding)
    embedder = BgeM3Embedder(embedding)
    path = Path(tempfile.mkdtemp(prefix="retold-embedding-calibration-")) / "memory.sqlite"
    count = build_fixture(path, embedder, config=config)
    store = Store(path)
    index = VectorIndex(embedding)
    index.load(store)
    labelled = _labelled()

    relevant: list[tuple[float, str, int]] = []
    for item in labelled["relevant"]:
        agent, user = item["principal"]
        eligible = store.eligible_ids(readable_scopes(store, agent, user), None, None, None, False, NOW)
        hits = index.search(embedder.embed_queries([item["query"]])[0], eligible, 50)
        ranked = [record_id for record_id, _ in hits]
        expected = item["expected"][0]
        record = store.get_record(expected)
        if record is None:
            raise RuntimeError(f"Missing expected record {expected}.")
        score = dict(hits).get(expected, 0.0)
        rank = ranked.index(expected) + 1 if expected in ranked else 0
        relevant.append((score, record.type, rank))

    irrelevant: list[tuple[float, str]] = []
    for item in labelled["irrelevant"]:
        agent, user = item["principal"]
        eligible = store.eligible_ids(readable_scopes(store, agent, user), None, None, None, False, NOW)
        hits = index.search(embedder.embed_queries([item["query"]])[0], eligible, 1)
        if not hits:
            irrelevant.append((0.0, "semantic"))
            continue
        record = store.get_record(hits[0][0])
        if record is None:
            raise RuntimeError(f"Missing top record {hits[0][0]}.")
        irrelevant.append((hits[0][1], record.type))

    floors: dict[str, float] = {}
    print(f"model={args.model} dims={args.dims} records={count}")
    for memory_type in ("semantic", "episodic", "procedural"):
        positives = [score for score, kind, _ in relevant if kind == memory_type]
        negatives = [score for score, kind in irrelevant if kind == memory_type]
        rows: list[tuple[float, float, float, float, int, int]] = []
        for step in range(71):
            floor = round(step * 0.01, 2)
            tp = sum(score >= floor for score in positives)
            fp = sum(score >= floor for score in negatives)
            recall = tp / len(positives) if positives else 1.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append((floor, f1, precision, recall, tp, fp))
        best = max(rows, key=lambda row: (row[1], row[2], row[0]))
        floors[memory_type] = best[0]
        print(
            f"{memory_type}: floor={best[0]:.2f} f1={best[1]:.3f} precision={best[2]:.3f} "
            f"recall={best[3]:.3f} tp={best[4]}/{len(positives)} fp={best[5]}/{len(negatives)}"
        )

    recall_at_1 = sum(rank == 1 for _, _, rank in relevant) / len(relevant)
    recall_at_8 = sum(0 < rank <= 8 for _, _, rank in relevant) / len(relevant)
    passed = sum(score >= floors[kind] for score, kind, _ in relevant)
    false_positives = sum(score >= floors[kind] for score, kind in irrelevant)
    precision = passed / (passed + false_positives) if passed + false_positives else 0.0
    recall = passed / len(relevant)
    print(f"rank recall: r@1={recall_at_1:.3f} r@8={recall_at_8:.3f}")
    print(
        f"calibrated dense gate: precision={precision:.3f} recall={recall:.3f} "
        f"tp={passed}/{len(relevant)} fp={false_positives}/{len(irrelevant)}"
    )
    dense_floors = DenseFloorConfig(
        semantic=floors["semantic"],
        episodic=floors["episodic"],
        procedural=floors["procedural"],
        session_summary=floors["episodic"],
    )
    calibrated = replace(
        config,
        retrieval=replace(config.retrieval, gate=replace(config.retrieval.gate, dense_floor=dense_floors)),
    )
    runtime = build_runtime(calibrated, store, embedder=embedder)
    relevant_found = 0
    for item in labelled["relevant"]:
        agent, user = item["principal"]
        response = runtime.retriever.search(
            Principal(agent, user, None, None),
            SearchRequest(
                queries=[item["query"]],
                context=None,
                types=None,
                entities=None,
                since=None,
                until=None,
                k=8,
                include_history=False,
            ),
        )
        returned = {result.record.id for result in response.results}
        relevant_found += item["expected"][0] in returned
    irrelevant_empty = 0
    for item in labelled["irrelevant"]:
        agent, user = item["principal"]
        response = runtime.retriever.search(
            Principal(agent, user, None, None),
            SearchRequest(
                queries=[item["query"]],
                context=None,
                types=None,
                entities=None,
                since=None,
                until=None,
                k=8,
                include_history=False,
            ),
        )
        irrelevant_empty += not response.results
    print(
        f"full retrieval: relevant={relevant_found}/{len(labelled['relevant'])} "
        f"irrelevant_empty={irrelevant_empty}/{len(labelled['irrelevant'])}"
    )
    print(f"fixture={path}")
    store.close()


if __name__ == "__main__":
    main()
