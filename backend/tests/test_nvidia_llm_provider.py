"""Tests for the NVIDIA LLM provider.

No network is touched: ``urllib.request.urlopen`` is patched, the same
technique ``test_nvidia_embedding_provider.py`` uses. What is verified:

- the provider satisfies the ``LLMProvider`` Protocol,
- missing credentials refuse rather than improvise,
- HTTP and transport failures surface as :class:`NvidiaLLMError`,
- the reply parser extracts exactly the first JSON object and nothing else,
- every action keyword maps to the right :class:`LLMAction`, and an
  unrecognised action degrades to a refusal rather than an exception.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from verity.engine.models import Chunk, MediaType
from verity.llm.provider import FakeProvider, LLMProvider
from verity.llm.nvidia_provider import (
    DEFAULT_LLM_MODEL,
    NvidiaLLMError,
    NvidiaLLMProvider,
)


class FakeResponse:
    """A context manager returning one canned completion body."""

    def __init__(self, body: dict) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._body).encode("utf-8")


def _chunk(text: str = "Notice period is 30 days.") -> Chunk:
    return Chunk(
        rel_path="handbook.txt",
        media_type=MediaType.TEXT,
        location="text line 3",
        ordinal=1,
        text=text,
    )


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def _provider() -> NvidiaLLMProvider:
    return NvidiaLLMProvider(
        api_key="test-key", base_url="https://example.com", model_id="test-model"
    )


def _decide_with_content(content: str, chunks: list[Chunk] | None = None) -> object:
    with patch(
        "urllib.request.urlopen", return_value=FakeResponse(_completion(content))
    ):
        return _provider().decide(
            messages=[{"role": "user", "content": "What is the notice period?"}],
            chunks=chunks if chunks is not None else [_chunk()],
            tools=["search"],
        )


def test_provider_satisfies_the_protocol() -> None:
    """The real provider goes through the same door as the test fake."""
    assert isinstance(_provider(), LLMProvider)


def test_missing_api_key_raises_instead_of_improvising() -> None:
    provider = NvidiaLLMProvider(api_key="", base_url="https://example.com")
    with pytest.raises(NvidiaLLMError, match="NVIDIA_API_KEY is not set"):
        provider.decide(messages=[{"role": "user", "content": "q"}], chunks=[], tools=[])


def test_missing_base_url_raises() -> None:
    provider = NvidiaLLMProvider(api_key="k", base_url="")
    with pytest.raises(NvidiaLLMError, match="NVIDIA_API_BASE_URL is not set"):
        provider.decide(messages=[{"role": "user", "content": "q"}], chunks=[], tools=[])


def test_http_error_is_a_provider_error() -> None:
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            "https://example.com/chat/completions", 401, "Unauthorized", {}, None
        ),
    ):
        with pytest.raises(NvidiaLLMError, match="HTTP 401"):
            _provider().decide(
                messages=[{"role": "user", "content": "q"}], chunks=[], tools=[]
            )


def test_empty_completion_is_an_error() -> None:
    with patch("urllib.request.urlopen", return_value=FakeResponse(_completion("   "))):
        with pytest.raises(NvidiaLLMError, match="empty completion"):
            _provider().decide(
                messages=[{"role": "user", "content": "q"}], chunks=[], tools=[]
            )


def test_bare_json_answer_is_parsed() -> None:
    action = _decide_with_content(
        '{"action": "answer", "text": "It is 30 days.", "citations": [1]}'
    )
    assert action.type == "answer"
    assert action.text == "It is 30 days."
    assert action.chunk_ids == [1]


def test_json_in_markdown_fences_is_parsed() -> None:
    action = _decide_with_content(
        '```json\n{"action": "answer", "text": "30 days.", "citations": [1]}\n```'
    )
    assert action.type == "answer"
    assert action.chunk_ids == [1]


def test_json_surrounded_by_prose_yields_the_first_object() -> None:
    action = _decide_with_content(
        'Sure. {"action": "answer", "text": "30 days.", "citations": [1]} Thanks!'
    )
    assert action.type == "answer"


def test_braces_inside_strings_do_not_confuse_the_parser() -> None:
    action = _decide_with_content(
        '{"action": "answer", "text": "the contract says {ok} period.", "citations": [2]}'
    )
    assert action.type == "answer"
    assert action.text == "the contract says {ok} period."
    assert action.chunk_ids == [2]


def test_each_action_keyword_maps_to_its_action() -> None:
    cases = [
        ('{"action": "search", "query": "notice period termination"}', "search"),
        (
            '{"action": "clarification", "question": "Which contract?"}',
            "clarification",
        ),
        ('{"action": "refusal", "reason": "not in the extracts"}', "refusal"),
    ]
    for content, expected in cases:
        action = _decide_with_content(content, chunks=[])
        assert action.type == expected, content
        if expected == "search":
            assert action.query == "notice period termination"
        elif expected == "clarification":
            assert action.question == "Which contract?"
        else:
            assert action.reason == "not in the extracts"


def test_an_unrecognized_action_degrades_to_refusal() -> None:
    """A model inventing an action keyword is a diagnosable failure, not a 500."""
    action = _decide_with_content('{"action": "summarize", "text": "..."}')
    assert action.type == "refusal"
    assert "summarize" in action.reason


def test_none_are_never_treated_as_citations() -> None:
    action = _decide_with_content(
        '{"action": "answer", "text": "x", "citations": [1, null, "two"]}'
    )
    assert action.chunk_ids == [1]


def test_no_json_in_the_reply_is_an_error() -> None:
    with pytest.raises(NvidiaLLMError, match="no JSON object"):
        _decide_with_content("The answer is thirty days.")


def test_a_reply_without_an_object_is_an_error() -> None:
    """A bare list contains no braces, so it is rejected as "no JSON object"."""
    with pytest.raises(NvidiaLLMError, match="no JSON object"):
        _decide_with_content('[1, 2, 3]')


def test_model_id_default_is_recorded() -> None:
    provider = NvidiaLLMProvider(api_key="k", base_url="https://example.com")
    assert provider.model_id  # a non-empty default, overridable via env


def test_an_empty_model_env_falls_back_to_the_default(monkeypatch) -> None:
    """.env.example ships the name with no value; that must mean "unset"."""
    monkeypatch.setenv("NVIDIA_LLM_MODEL", "   ")
    provider = NvidiaLLMProvider(api_key="k", base_url="https://example.com")
    assert provider.model_id == DEFAULT_LLM_MODEL


def test_chunks_are_numbered_in_the_prompt_by_position() -> None:
    """The citation contract needs position 1 to mean the first shown chunk."""
    provider = _provider()
    section = provider._chunks_section([_chunk("first"), _chunk("second")])
    assert section.startswith("Document extracts this turn")
    assert "[1] (handbook.txt, text line 3):" in section


def test_empty_chunk_list_says_so() -> None:
    provider = _provider()
    section = provider._chunks_section([])
    assert "none this turn" in section


def test_fake_provider_still_satisfies_the_protocol() -> None:
    assert isinstance(FakeProvider(), LLMProvider)
