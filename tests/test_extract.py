"""Pulling the graded artifact out of a reply: fenced JSON blocks, closed or left open at the end."""

from genbench.grade.extract import extract
from genbench.pack.models import Extract

JSON = Extract(kind="json_block")


def test_a_closed_json_block_is_taken_as_is():
    art = extract('Here:\n```json\n{"a": 1}\n```\nDone.', JSON)
    assert (art.text, art.found_block, art.closed) == ('{"a": 1}', True, True)


def test_a_json_block_left_open_at_the_end_is_still_graded():
    art = extract('```json\n{"a": 1}', JSON)
    assert (art.text, art.found_block, art.closed) == ('{"a": 1}', True, False)


def test_an_untagged_open_fence_counts_too():
    assert extract('```\n{"a": 1}\n', JSON).text == '{"a": 1}'


def test_a_closed_block_wins_over_a_later_open_one():
    art = extract('```json\n{"a": 1}\n```\nAnd again:\n```json\n{"a": 2', JSON)
    assert (art.text, art.closed) == ('{"a": 1}', True)


def test_the_open_block_is_found_after_thinking_is_removed():
    art = extract('<think>maybe ```json then</think>\n```json\n{"a": 1}', JSON)
    assert (art.text, art.closed) == ('{"a": 1}', False)


def test_an_open_fence_with_nothing_after_it_falls_back_to_the_whole_reply():
    art = extract("```json\n   ", JSON)
    assert (art.text, art.found_block) == ("```json", False)


def test_an_open_code_fence_is_not_salvaged():
    """An unclosed code fence is usually a reply cut off mid-file, so code tests keep the old rule."""
    art = extract("```python\nprint(1)\n", Extract(kind="code_block", langs=["python"]))
    assert (art.found_block, art.closed) == (False, True)
