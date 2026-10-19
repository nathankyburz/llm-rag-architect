"""Corpus manifest: which PDFs to ingest, and the metadata to attach."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag_architect.domain.documents import DocumentType


class DocumentSpec(BaseModel):
    """One entry of the manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    document_type: DocumentType
    title: str = Field(min_length=1)
    club: str | None = Field(default=None)
    fiscal_year: int | None = Field(default=None, ge=1900, le=2100)


def load_manifest(path: Path, *, root: Path) -> list[DocumentSpec]:
    """
    Read the manifest and resolve each entry's path against `root`.

    Fails on a missing file rather than skipping it: a typo in the manifest
    would otherwise shrink the corpus silently, and nothing downstream could
    tell the difference between "not listed" and "listed but never read".
    """
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    specs = [DocumentSpec.model_validate(entry) for entry in raw.get("documents", [])]
    resolved = [spec.model_copy(update={"path": root / spec.path}) for spec in specs]

    missing = [spec.path for spec in resolved if not spec.path.is_file()]
    if missing:
        listed = ", ".join(str(p) for p in missing)
        msg = f"manifest lists files that do not exist: {listed}"
        raise FileNotFoundError(msg)
    return resolved
