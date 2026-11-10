"""The built-in core pack: the four original tasks, ported 1:1."""

import ast
import json
from pathlib import Path

import pytest

from genbench.config import PassMark, load_config
from genbench.pack.loader import BUILTIN_DIR, Catalog, load_catalog
from genbench.pack.validate import validate_catalog
from genbench.schema import pack_schema

ROOT = Path(__file__).resolve().parents[1]
OLD_SCRIPT = ROOT.parent / "ollama_local_benchmark.py"
IDS = ["python_streamlit_job_parser", "bash_network_monitor", "ai_gateway_architecture", "algo_and_tool_dispatch"]
OLD_IDS = ["python_coding", "bash_coding", "system_architecture", "algo_and_agentic"]


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(None, env={})
    everything = load_catalog(cfg)
    return cfg, Catalog(packs=[lp for lp in everything.packs if lp.pack.pack == "core"], issues=everything.issues)


def test_every_shipped_pack_loads_with_no_config():
    catalog = load_catalog(load_config(None, env={}))
    assert [lp.pack.pack for lp in catalog.packs] == ["core", "code-reasoning", "long-context", "stretch"]
    assert len(catalog.tests) == 14 and catalog.issues == [] and all(lp.builtin for lp in catalog.packs)


def test_core_pack_loads_cleanly(setup):
    cfg, catalog = setup
    assert [lp.pack.pack for lp in catalog.packs] == ["core"]
    assert [lt.test.id for lt in catalog.tests] == IDS
    errors = [i for i in validate_catalog(catalog, cfg, portable=True) if i.level == "error"]
    assert errors == []


def test_core_pack_is_portable_to_rust(setup):
    cfg, catalog = setup
    portability = [i for i in validate_catalog(catalog, cfg, portable=True) if "Rust" in i.message]
    assert portability == []


def test_every_test_explains_itself(setup):
    _, catalog = setup
    for lt in catalog.tests:
        test = lt.test
        assert test.name and test.description and test.goal
        for _, checks in test.rubric.sections():
            assert all(check.expect for check in checks)


def test_rubrics_keep_the_old_100_point_scales(setup):
    _, catalog = setup
    for lt in catalog.tests:
        assert lt.test.rubric.max_score("accuracy") == 100, lt.test.id
        if lt.test.rubric.quality != "same_as_accuracy":
            assert lt.test.rubric.max_score("quality") == 100, lt.test.id
    assert catalog.find("algo_and_tool_dispatch").test.rubric.quality == "same_as_accuracy"


def test_core_uses_the_configured_pass_mark(setup):
    cfg, catalog = setup
    stricter = PassMark(accuracy=85)
    for lt in catalog.tests:
        assert lt.test.pass_mark(cfg.scoring.pass_mark).accuracy == 70
        assert lt.test.pass_mark(stricter) is stricter


def test_old_task_names_are_aliases(setup):
    _, catalog = setup
    assert catalog.find("1. Python Coding (Streamlit HTML Job Parser)").test.id == IDS[0]
    assert catalog.find("4. Algorithmic & Agentic Triage (Complexity + JSON Tool Dispatch)").test.id == IDS[3]


def test_prompts_match_the_original_script_exactly(setup):
    if not OLD_SCRIPT.is_file():
        pytest.skip("ollama_local_benchmark.py not next to genbench/")
    tree = ast.parse(OLD_SCRIPT.read_text(encoding="utf-8"))
    tasks = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "TASKS"
    )
    old = {t["id"]: t for t in tasks}
    _, catalog = setup
    for new_id, old_id in zip(IDS, OLD_IDS, strict=True):
        lt = catalog.find(new_id)
        assert lt.prompt_text() == old[old_id]["prompt"], new_id
        assert lt.test.category == old[old_id]["category"]
        assert old[old_id]["name"] in lt.test.aliases


def test_packaged_schema_is_current():
    stored = json.loads((BUILTIN_DIR.parent / "schemas" / "testpack.schema.json").read_text(encoding="utf-8"))
    assert stored == pack_schema(), "run: genbench schema --out src/genbench/schemas/testpack.schema.json"


def test_project_schema_copy_is_current():
    stored = json.loads((ROOT / "schemas" / "testpack.schema.json").read_text(encoding="utf-8"))
    assert stored == pack_schema(), "run: genbench schema --out schemas/testpack.schema.json"


def test_core_pack_passes_the_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    data = json.loads((BUILTIN_DIR / "core" / "pack.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(pack_schema()).validate(data)


def test_json_schema_requires_expect_on_top_level_checks():
    jsonschema = pytest.importorskip("jsonschema")
    from conftest import minimal_test

    bad = minimal_test()
    del bad["rubric"]["accuracy"][0]["expect"]
    pack = {"pack": "x", "version": "1.0.0", "tests": [bad]}
    with pytest.raises(jsonschema.ValidationError, match="expect"):
        jsonschema.Draft202012Validator(pack_schema()).validate(pack)
