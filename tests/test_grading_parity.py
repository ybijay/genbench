"""genbench's core pack must score exactly like the original script's grader functions.

Every saved answer is graded both ways, together with altered versions that hit the edge cases:
missing fences, broken syntax, think tags, missing requirements, odd JSON and empty answers.
"""

import os
import re
import shutil

import pytest
from legacy_ref import old_grade, old_module, saved_outputs

from genbench.config import load_config
from genbench.grade.engine import GradeContext, grade_test
from genbench.grade.validators import ValidatorRunner
from genbench.pack.loader import load_catalog

pytestmark = pytest.mark.skipif(old_module() is None, reason="ollama_local_benchmark.py not next to genbench/")
if os.name != "nt" and shutil.which("bash") is None:
    pytestmark = pytest.mark.skip(reason="the old grader needs bash on this platform")

CFG = load_config(None, env={})
CATALOG = load_catalog(CFG)

GOOD_PY = '''```python
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup

if "html" not in st.session_state:
    st.session_state.html = ""
html = st.text_area("Paste HTML", key="html")
if st.button("Clear"):
    st.session_state.html = ""
    st.rerun()
try:
    soup = BeautifulSoup(html, "html.parser")
    rows = [{"title": t.get_text()} for t in soup.find_all("h2")]
    st.dataframe(pd.DataFrame(rows))
except Exception as exc:
    st.error(str(exc))
```'''
GOOD_SH = '''Here is the script:

```bash
#!/bin/bash
IP="192.168.1.20"
FAILS=0
while true; do
  if ping -c 1 "$IP" > /dev/null; then
    T=$(curl -s -o /dev/null -w "%{time_total}" "http://$IP:80")
    echo "$(date),$T" >> log.csv
    FAILS=0
  else
    FAILS=$((FAILS+1))
  fi
  if [ "$FAILS" -ge 3 ]; then
    notify-send "Device down"
  fi
  sleep 10
done
```'''
GOOD_ARCH = ("A gateway with a FIFO queue and a semaphore protects 24 GB of VRAM from OOM. KV-cache pinning "
             "versus hot-swap, TTFT with prompt caching and streaming, health checks with a circuit breaker.\n\n"
             "```\n[IDE] --> [Gateway] --> [Ollama]\n```\n\n| Option | Pro | Con |\n|---|---|---|\n| Pin | fast | memory |\n"
             + " ".join(["detail"] * 320))
GOOD_AGENT = ('QuickSort averages O(n log n) and degrades to O(n^2); MergeSort is O(n log n) with O(n) auxiliary '
              'space. Random or median-of-three pivots help.\n\n```json\n[{"name": "get_container_logs", "arguments": '
              '{"service_name": "checkout-service", "node_id": "prod-us-east-04", "tail_lines": 200}},\n {"name": '
              '"get_node_metrics", "arguments": {"node_id": "prod-us-east-04", "metric_types": ["memory"]}},\n {"name": '
              '"get_deployment_history", "arguments": {"service_name": "checkout-service", "limit": 5}}]\n```')

HAND_CASES = {
    "python_streamlit_job_parser": [GOOD_PY, "", "I can't help with that.", "```python\n```", GOOD_PY.replace("```python", "```py"),
                                    "<think>plan it</think>" + GOOD_PY, GOOD_PY + "\n\n```python\nprint(1)\n```"],
    "bash_network_monitor": [GOOD_SH, "", "No script.", GOOD_SH.replace("```bash", "```sh"), GOOD_SH.replace("```bash", "```"),
                             GOOD_SH.replace("fi\n  sleep", "\n  sleep")],
    "ai_gateway_architecture": [GOOD_ARCH, "", "Use a queue.", GOOD_ARCH.upper(), "<think>x</think>" + GOOD_ARCH[:400]],
    "algo_and_tool_dispatch": [
        GOOD_AGENT, "",
        GOOD_AGENT.replace("```json\n[", '```json\n{"tool_calls": [').replace("]\n```", "]}\n```"),
        GOOD_AGENT.replace("prod-us-east-04", "prod-us-east-05"),
        GOOD_AGENT.replace("```json\n", "").replace("\n```", ""),
        GOOD_AGENT.replace('"arguments": {"node_id": "prod-us-east-04", "metric_types": ["memory"]}',
                           '"arguments": "{\\"node_id\\": \\"prod-us-east-04\\"}"'),
        GOOD_AGENT.replace("]\n```", ', {"name": "get_node_metrics", "arguments": {"node_id": "prod-us-east-04"}}]\n```'),
        GOOD_AGENT.replace("```json", "```json\n{not json}\n```\n\n```json"),
        "o(n log n) O(N^2) MERGE SORT median of three in-place",
    ],
}


def variants(test_id: str, text: str) -> list[tuple[str, str]]:
    """Altered copies of a saved answer that exercise the grader's branches."""
    out = [("original", text), ("think-tags", "<think>first I plan</think>\n" + text)]
    if test_id == "python_streamlit_job_parser":
        out += [
            ("no-try", text.replace("try:", "if True:")),
            ("syntax-error", re.sub(r"\bdef ", "def def ", text, count=1)),
            ("untagged-fence", text.replace("```python", "```")),
            ("no-fences", text.replace("```python", "").replace("```", "")),
            ("no-dataframe", text.replace("st.dataframe", "st.write").replace("st.table", "st.write")),
            ("styling-hack", text.replace("st.title(", 'st.markdown("<style>x</style>", unsafe_allow_html=True)\nst.title(', 1)),
            ("no-clear", text.replace("session_state", "state").replace("rerun", "go").replace("Clear", "Go").replace("clear", "go")),
        ]
    elif test_id == "bash_network_monitor":
        out += [
            ("no-shebang", re.sub(r"(?m)^#!.*\n", "", text)),
            ("no-notify", text.replace("notify-send", "echo")),
            ("unclosed-if", re.sub(r"(?m)^\s*fi\s*$", "", text, count=1)),
            ("sh-fence", text.replace("```bash", "```sh")),
            ("untagged-fence", text.replace("```bash", "```")),
            ("no-loop", text.replace("while ", "if ").replace("sleep", "wait")),
        ]
    elif test_id == "ai_gateway_architecture":
        out += [
            ("short", " ".join(text.split()[:120])),
            ("no-pipes", text.replace("|", "/")),
            ("no-arrows", text.replace("-->", "to").replace("==>", "to").replace("+---", "+").replace("|---", "|")),
            ("lowercase", text.lower()),
        ]
    else:
        out += [
            ("no-json-fence", text.replace("```json", "```")),
            ("wrong-node", text.replace("prod-us-east-04", "prod-us-east-99")),
            ("no-mergesort", re.sub(r"(?i)merge ?sort", "the other sort", text)),
            ("upper", text.upper()),
        ]
    return out


CASES = []
for _test_id, _label, _output in saved_outputs():
    for _name, _text in variants(_test_id, _output):
        CASES.append(pytest.param(_test_id, _text, id=f"{_test_id}:{_label}:{_name}"))
for _test_id, _texts in HAND_CASES.items():
    for _i, _text in enumerate(_texts):
        CASES.append(pytest.param(_test_id, _text, id=f"{_test_id}:hand{_i}"))


@pytest.mark.parametrize(("test_id", "output"), CASES)
def test_same_scores_as_the_original_grader(test_id, output, tmp_path):
    lt = CATALOG.find(test_id)
    force = {"bash"} if os.name == "nt" else set()  # the old script used its heuristic on Windows
    ctx = GradeContext(ValidatorRunner(CFG.validators, force), tmp_path / "artifacts", lt.pack.dir, CFG.scoring.pass_mark)
    grade = grade_test(lt.test, output, ctx)
    old_accuracy, old_score = old_grade(test_id, output)
    assert (grade.accuracy, grade.quality) == (pytest.approx(old_accuracy, abs=1e-9), pytest.approx(old_score, abs=1e-9))


def test_there_are_enough_cases():
    assert len(CASES) >= 30
