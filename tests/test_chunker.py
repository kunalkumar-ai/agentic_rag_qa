from chunker import build_chunks, parse_sections

SAMPLE_TEXT = """ITEM 1. BUSINESS
Overview
We design and manufacture electric vehicles and energy storage systems.

We sell them directly to customers through our website and stores.

ITEM 1A. RISK FACTORS
We may experience delays in launching products.

We face significant competition from other manufacturers."""


def test_parse_sections_finds_two_items():
    sections = parse_sections(SAMPLE_TEXT)
    assert len(sections) == 2


def test_parse_sections_names_start_with_item():
    sections = parse_sections(SAMPLE_TEXT)
    assert sections[0]["name"].startswith("ITEM 1.")
    assert sections[1]["name"].startswith("ITEM 1A.")


def test_each_section_has_text():
    sections = parse_sections(SAMPLE_TEXT)
    for section in sections:
        assert len(section["text"].strip()) > 0


def test_build_chunks_produces_parents_and_children():
    chunks = build_chunks(SAMPLE_TEXT)
    types = {c.chunk_type for c in chunks}
    assert "parent" in types
    assert "child" in types


def test_child_chunk_parent_id_matches_a_parent():
    chunks = build_chunks(SAMPLE_TEXT)
    parent_ids = {c.chunk_id for c in chunks if c.chunk_type == "parent"}
    children = [c for c in chunks if c.chunk_type == "child"]
    for child in children:
        assert child.parent_id in parent_ids


def test_no_empty_chunks():
    chunks = build_chunks(SAMPLE_TEXT)
    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_section_name_propagated_to_children():
    chunks = build_chunks(SAMPLE_TEXT)
    children = [c for c in chunks if c.chunk_type == "child"]
    for child in children:
        assert child.section_name != ""


def test_very_short_paragraphs_are_skipped():
    text = """ITEM 2. PROPERTIES
We own the following properties across our global operations.

Note

These properties support manufacturing operations across all regions."""
    chunks = build_chunks(text)
    children = [c for c in chunks if c.chunk_type == "child"]
    child_texts = [c.text for c in children]
    assert not any(t.strip() == "Note" for t in child_texts)


def test_chunk_ids_are_unique():
    chunks = build_chunks(SAMPLE_TEXT)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_company_and_year_tagged_on_every_chunk():
    chunks = build_chunks(SAMPLE_TEXT, company="tesla", year="2024")
    for chunk in chunks:
        assert chunk.company == "tesla"
        assert chunk.year == "2024"


def test_chunk_id_prefixed_with_company_and_year():
    chunks = build_chunks(SAMPLE_TEXT, company="gm", year="2023")
    for chunk in chunks:
        assert chunk.chunk_id.startswith("gm_2023_")
