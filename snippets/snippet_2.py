"""
Pattern 9 · Index-Aware Retrieval — Episodio 4
Trazabilidad con etiquetas [chunk:42]: el linaje que sostiene la auditoría.

Idea central: cada claim verificada debe llevar el "expediente" del chunk
que la sostiene. Sin un chunk_id estable que viaje desde la indexación
hasta el veredicto, no hay auditoría posible.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Literal, Sequence

# ──────────────────────────────────────────────────────────────────────────────
# 1. Familia de metadata pre-retrieval: el contrato atómico chunk ↔ metadata
# ──────────────────────────────────────────────────────────────────────────────

NLI_LABEL = Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]


@dataclass(frozen=True)
class Metadata:
    created_at: datetime
    source: str
    author: str
    category: str
    version: str


@dataclass(frozen=True)
class Chunk:
    """Trocito textual con su trío inseparable: chunk + embedding (implícito) + metadata."""

    text: str
    metadata: Metadata
    # id estable: en producción usa ULID o UUIDv5 sobre (source, version, text).
    # Aquí basta con un digest corto para que el ejemplo sea legible.
    id: str = field(init=False)

    def __post_init__(self) -> None:
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:8]
        object.__setattr__(self, "id", f"chunk:{digest}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Lo que sale del retriever (ya con la valla τ(q) y la rodilla aplicadas)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    cosine_score: float  # ya pasado el filtro adaptativo del Ep.2/3


# ──────────────────────────────────────────────────────────────────────────────
# 3. Claim + veredicto: aquí nace la etiqueta [chunk:42]
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Claim:
    text: str
    origin: str  # p. ej. "llm_draft" o "user_assertion"


@dataclass(frozen=True)
class VerificationResult:
    claim: Claim
    label: NLI_LABEL
    nli_score: float
    supporting_chunk_id: str | None  # expediente
    citation_tag: str               # etiqueta visible para el lector
    anchored: bool                  # True ⇔ label == ENTAILMENT y score ≥ τ
    rationale: str


# ──────────────────────────────────────────────────────────────────────────────
# 4. Verificador: NLI frase-por-frase + anclaje al chunk superviviente
# ──────────────────────────────────────────────────────────────────────────────

# τ mínimo de entailment para declarar una claim "anclada".
ANCHOR_THRESHOLD = 0.70


def _nli_score(premise: str, hypothesis: str) -> tuple[NLI_LABEL, float]:
    """
    Devuelve (label, score) para (premise ⊨ hypothesis).

    Implementación real (descomentar si tienes `transformers` instalado):

        from transformers import pipeline
        nli = pipeline(
            "text-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            top_k=1,
        )
        out = nli(f"{premise} [SEP] {hypothesis}")[0][0]
        label = {"ENTAILMENT": "ENTAILMENT",
                 "NEUTRAL": "NEUTRAL",
                 "CONTRADICTION": "CONTRADICTION"}[out["label"]]
        return label, float(out["score"])
    """
    # Fallback determinístico basado en solapamiento léxico (suficiente para demo).
    p_tokens = set(re.findall(r"\w+", premise.lower()))
    h_tokens = set(re.findall(r"\w+", hypothesis.lower()))
    if not p_tokens or not h_tokens:
        return "NEUTRAL", 0.0
    overlap = len(p_tokens & h_tokens) / len(p_tokens | h_tokens)
    if overlap >= 0.55:
        return "ENTAILMENT", min(1.0, overlap + 0.2)
    if overlap <= 0.10:
        return "CONTRADICTION", min(1.0, 0.6 + (0.10 - overlap))
    return "NEUTRAL", overlap


class ClaimVerifier:
    """Asocia cada claim al chunk superviviente que mejor la sostiene."""

    def __init__(self, threshold: float = ANCHOR_THRESHOLD) -> None:
        self.threshold = threshold

    def verify(self, claim: Claim, retrieved: Sequence[RetrievedChunk]) -> VerificationResult:
        if not retrieved:
            return VerificationResult(
                claim=claim,
                label="NEUTRAL",
                nli_score=0.0,
                supporting_chunk_id=None,
                citation_tag="[unsupported]",
                anchored=False,
                rationale="Sin chunks supervivientes tras el filtro adaptativo.",
            )

        # NLI frase-por-frase contra cada chunk; nos quedamos con el mejor entailment.
        best_label: NLI_LABEL = "NEUTRAL"
        best_score = -1.0
        best_chunk_id: str | None = None

        for rc in retrieved:
            label, score = _nli_score(rc.chunk.text, claim.text)
            # Empate: preferimos entailment > contradiction > neutral.
            rank = {"ENTAILMENT": 2, "CONTRADICTION": 1, "NEUTRAL": 0}
            if (rank[label], score) > (rank[best_label], best_score):
                best_label, best_score, best_chunk_id = label, score, rc.chunk.id

        anchored = best_label == "ENTAILMENT" and best_score >= self.threshold
        citation = f"[{best_chunk_id}]" if anchored else "[unsupported]"
        rationale = (
            f"Mejor match: {best_chunk_id} con {best_label}={best_score:.2f}"
            + (" ≥ τ → anclada." if anchored else " < τ o no-ENTAILMENT → no anclada.")
        )

        return VerificationResult(
            claim=claim,
            label=best_label,
            nli_score=best_score,
            supporting_chunk_id=best_chunk_id if anchored else None,
            citation_tag=citation,
            anchored=anchored,
            rationale=rationale,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Demostración: una claim verdadera y una falsa, contra 4 chunks
# ──────────────────────────────────────────────────────────────────────────────

def _mk_chunk(text: str, source: str, author: str, category: str, version: str) -> Chunk:
    return Chunk(
        text=text,
        metadata=Metadata(
            created_at=datetime.now(timezone.utc),
            source=source,
            author=author,
            category=category,
            version=version,
        ),
    )


def main() -> None:
    corpus: Iterable[Chunk] = [
        _mk_chunk(
            "Pgvector permite almacenar embeddings y ejecutar búsqueda por similitud "
            "coseno dentro de PostgreSQL.",
            source="docs/pgvector.md",
            author="platform-team",
            category="vector-store",
            version="0.7.3",
        ),
        _mk_chunk(
            "FAISS es una biblioteca de Facebook para búsqueda de vecinos más cercanos "
            "en espacios densos de alta dimensión.",
            source="docs/faiss.md",
            author="search-team",
            category="vector-store",
            version="1.0",
        ),
        _mk_chunk(
            "El modelo atómico chunk↔metadata exige que toda metadata viaje con el "
            "chunk y nunca se infiera a posteriori.",
            source="internal/pattern9.md",
            author="ragschool",
            category="methodology",
            version="ep2",
        ),
        _mk_chunk(
            "La valla τ(q) = μ + k·σ es adaptativa y se recalibra con ground truth.",
            source="internal/pattern9.md",
            author="ragschool",
            category="methodology",
            version="ep2",
        ),
    ]
    chunks = list(corpus)

    # Simula la salida del retriever tras filtro + valla (Ep.2/3).
    retrieved: list[RetrievedChunk] = [
        RetrievedChunk(chunk=chunks[0], cosine_score=0.91),
        RetrievedChunk(chunk=chunks[1], cosine_score=0.78),
        RetrievedChunk(chunk=chunks[2], cosine_score=0.66),
        RetrievedChunk(chunk=chunks[3], cosine_score=0.61),
    ]

    verifier = ClaimVerifier(threshold=ANCHOR_THRESHOLD)

    claims = [
        Claim(text="pgvector hace búsqueda por coseno en Postgres.", origin="llm_draft"),
        Claim(text="FAISS fue creado por Google.", origin="llm_draft"),
    ]

    for c in claims:
        result = verifier.verify(c, retrieved)
        print("─" * 72)
        print(f"CLAIM : {c.text}")
        print(f"LABEL : {result.label}  (score={result.nli_score:.2f})")
        print(f"ANCLA : {result.citation_tag}  anchored={result.anchored}")
        print(f"POR QUÉ: {result.rationale}")


if __name__ == "__main__":
    main()