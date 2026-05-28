"""Prompt enrichment + tag extraction. Pure-Python — runs without torch."""

from himat.data import prompts as P


def test_enrich_template_with_tags():
    out = P.enrich_template("Wood", ["oak", "planks", "rough"])
    assert "wood" in out
    assert "oak" in out and "planks" in out
    assert "tileable" in out.lower()


def test_enrich_template_no_tags():
    out = P.enrich_template("Metal", [])
    assert "metal" in out
    assert "tileable" in out.lower()


def test_enrich_template_handles_none():
    out = P.enrich_template("", None)
    assert isinstance(out, str) and out


def test_wrap_for_inference_prepends_system_prompt():
    wrapped = P.wrap_for_inference("a rusty metal surface")
    assert wrapped.startswith(P.SYSTEM_PROMPT)
    assert "rusty metal" in wrapped


def test_build_llm_prompt_mentions_attributes():
    instr = P.build_llm_prompt("Stone", ["grey", "cracked"])
    # the four attributes the paper names should be requested
    for word in ("colour", "roughness", "imperfections"):
        assert word in instr.lower()
    assert "Stone" in instr and "cracked" in instr


def test_extract_tags_nested_metadata():
    rec = {
        "category": "Fabric",
        "metadata": {"tags": ["red", "woven"], "description": "a soft woven textile"},
    }
    cat, tags = P.extract_tags(rec)
    assert cat == "Fabric"
    # description is inserted first, then the tags
    assert tags[0] == "a soft woven textile"
    assert "red" in tags and "woven" in tags


def test_extract_tags_string_tags():
    rec = {"category": "Ground", "metadata": {"tags": "dirt; gravel, dry"}}
    cat, tags = P.extract_tags(rec)
    assert cat == "Ground"
    assert set(tags) == {"dirt", "gravel", "dry"}


def test_extract_tags_empty():
    cat, tags = P.extract_tags({})
    assert cat == "" and tags == []
