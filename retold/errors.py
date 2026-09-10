"""Exceptions that a host is expected to handle by name."""

from __future__ import annotations


class LocalModelUnavailable(RuntimeError):
    """A local model could not be imported or loaded.

    Raised by the embedder, the NLI judge, and the reranker when ``sentence-transformers`` is not installed
    or the model files are not in the local cache. It is the one failure a host fixes by installing the
    ``local-models`` extra or allowing the first download; every other ``RuntimeError`` means something else.
    """


class EmbeddingProfileMismatch(RuntimeError):
    """The store contains vectors produced by a different embedding profile."""

    def __init__(
        self,
        configured: tuple[str, str, int],
        stored: list[tuple[str, str, int]],
    ) -> None:
        configured_label = f"{configured[0]}/{configured[1]} ({configured[2]} dimensions)"
        stored_label = ", ".join(f"{model}/{version} ({dims} dimensions)" for model, version, dims in stored)
        super().__init__(
            f"The store contains embeddings from {stored_label}, but Retold is configured for {configured_label}. "
            "Recalibrate the retrieval floors, then run `retold reembed` with the new model configuration before "
            "opening this store."
        )


class UnsupportedEvidenceError(ValueError):
    """The quote does not support the claim, so the write would be stored as an expiring inference.

    Raised by the facade's ``remember`` for a ``user_statement`` or ``tool_result`` claim the evidence check
    downgraded. Pass ``allow_inference=True`` to accept the provisional record deliberately.
    """

    def __init__(self, claim: str, evidence: str, note: str | None) -> None:
        self.claim = claim
        self.evidence = evidence
        self.note = note
        reason = note or "the evidence does not support the claim"
        super().__init__(
            f"The quote {evidence!r} does not support {claim!r} ({reason}), so Retold would store it as an "
            "expiring inference. Phrase the content the way the quote says it, or pass allow_inference=True."
        )
