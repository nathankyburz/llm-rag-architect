import pytest
from pydantic import ValidationError

from rag_architect.domain.documents import DocumentType
from rag_architect.ingestion.manifest import load_manifest

MANIFEST = """
[[documents]]
path = "brighton24.pdf"
document_type = "club_accounts"
title = "Brighton FY2024"
club = "Brighton & Hove Albion"
fiscal_year = 2024

[[documents]]
path = "deloitte.pdf"
document_type = "industry_report"
title = "Deloitte 2026"
"""


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "brighton24.pdf").write_bytes(b"%PDF-")
    (tmp_path / "deloitte.pdf").write_bytes(b"%PDF-")
    return tmp_path


def write_manifest(tmp_path, body: str = MANIFEST):
    path = tmp_path / "manifest.toml"
    path.write_text(body)
    return path


def test_entries_are_parsed_and_typed(corpus):
    specs = load_manifest(write_manifest(corpus), root=corpus)

    assert len(specs) == 2
    assert specs[0].document_type is DocumentType.CLUB_ACCOUNTS
    assert specs[0].fiscal_year == 2024
    assert specs[1].document_type is DocumentType.INDUSTRY_REPORT
    assert specs[1].club is None


def test_paths_are_resolved_against_the_root(corpus):
    specs = load_manifest(write_manifest(corpus), root=corpus)

    assert specs[0].path == corpus / "brighton24.pdf"
    assert specs[0].path.is_file()


def test_a_listed_file_that_does_not_exist_is_an_error(corpus):
    """
    A typo must not silently shrink the corpus: nothing downstream could tell
    "not listed" from "listed but never read".
    """
    body = (
        MANIFEST
        + '\n[[documents]]\npath = "absent.pdf"\ndocument_type = "club_accounts"\ntitle = "x"\n'
    )

    with pytest.raises(FileNotFoundError, match=r"absent\.pdf"):
        load_manifest(write_manifest(corpus, body), root=corpus)


def test_an_unknown_key_is_rejected(corpus):
    body = (
        MANIFEST
        + '\n[[documents]]\npath = "brighton24.pdf"\n'
        + 'document_type = "club_accounts"\ntitle = "x"\nleague = "PL"\n'
    )

    with pytest.raises(ValidationError):
        load_manifest(write_manifest(corpus, body), root=corpus)


def test_an_unknown_document_type_is_rejected(corpus):
    body = '[[documents]]\npath = "brighton24.pdf"\ndocument_type = "press_release"\ntitle = "x"\n'

    with pytest.raises(ValidationError):
        load_manifest(write_manifest(corpus, body), root=corpus)


def test_an_empty_manifest_yields_nothing(corpus):
    assert load_manifest(write_manifest(corpus, ""), root=corpus) == []
