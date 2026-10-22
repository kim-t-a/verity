"""NVIDIA-backed LLM provider.

The real answering model, satisfying the same
:class:`~verity.llm.provider.LLMProvider` Protocol the tests exercise with
:class:`~verity.llm.provider.FakeProvider`. Standard-library HTTP only
(``urllib.request``), mirroring
:class:`~verity.engine.nvidia_embedding_provider.NvidiaEmbeddingProvider`, so the
product layer stays out of the engine's zero-dependency guarantee.

The decision protocol is structured JSON, not tool-calling: the system
prompt numbers the chunks it was shown and demands exactly one JSON object
with an ``action`` of ``answer`` / ``clarification`` / ``refusal`` /
``search``. The agent loop verifies any cited positions against the chunks
actually shown (invariant I2), so a model that hallucinates a position is
caught by a check in code, not by the model's own good intentions.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from verity.engine.models import Chunk
from verity.llm.provider import LLMAction

#: Default chat model. Verified on the configured key on 2026-10-22 with the
#: real provider path (one chunk, one question): answered with citation [1]
#: in 6.9 s. ``google/gemma-4-31b-it`` is still listed in the catalog but
#: timed out repeatedly on this account, so it is not the default.
DEFAULT_LLM_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

#: How many chunks are named in the system prompt. The agent loop retrieves
#: at most RETRIEVAL_LIMIT chunks per turn; showing all of them is the point
#: of invariant I2, so no additional cap is applied here.
#: Raised from 1024 with the plain-prose answer style: a natural explanation
#: of several passages ran into the old budget and was cut off mid-sentence
#: (``finish_reason=length``), which reads as a broken answer, not a shorter one.
MAX_RESPONSE_TOKENS = 2048

#: How many messages of the conversation the follow-up rewriter is shown.
#: Enough to resolve "it" or "the other one"; short enough that a long
#: conversation does not turn one search query into a large prompt.
REWRITE_HISTORY_MESSAGES = 6

#: The default model reasons before it writes. With ten retrieved chunks it
#: spent all 1024 tokens on ``reasoning_content`` and returned an empty
#: ``content``; 4096 and even 8192 tokens were no better at 1024's speed.
#: ``chat_template_kwargs`` turns that off. Verified against the real endpoint
#: on 2026-10-22: a ten-chunk prompt answered with citations in 4.4 s and zero
#: reasoning tokens. Set ``NVIDIA_LLM_DISABLE_THINKING=0`` if a configured
#: model rejects the parameter.
DISABLE_THINKING_DEFAULT = True


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment flag; an empty value means ``default``."""
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}

SYSTEM_PROMPT = """You answer questions strictly from the numbered document \
extracts shown to you. You reply in exactly one JSON object and never in any \
other format.

Choose one action:
- answer: the extracts contain the answer. Reply
  {"action": "answer", "text": "<the answer>", "citations": [<chunk numbers>]}
  Every sentence of borrowed fact must trace to a chunk number you cite.\
 Cite only numbers from the list above.
  Write it the way a knowledgeable colleague would say it out loud: plain \
sentences, no preamble, no restating of the question, no headings. Put the \
chunk number in square brackets immediately after the fact it supports, for \
example "Maximum junction temperature is 94 C [1]." Cite a chunk once per \
claim; a wall of bracketed numbers is not evidence a reader can follow.
  When the user asks you to explain, simplify or rephrase what you just said, \
say the same fact freshly and more plainly from the extracts shown this turn. \
Do not repeat your previous wording.
- clarification: the question is ambiguous and one short question would fix \
it. Reply {"action": "clarification", "question": "<one question>"}. Do not \
clarify when you can answer.
- refusal: the extracts genuinely do not contain the answer. Reply
  {"action": "refusal", "reason": "<what was searched and why it falls short>"}
- search: the question needs different wording before you can answer. Reply
  {"action": "search", "query": "<the rewritten query>"}

Never invent facts, documents, or chunk numbers.
"""

REWRITE_SYSTEM_PROMPT = """You turn one follow-up message into a standalone \
search query.

- Resolve what the message points at from the conversation: pronouns, "the \
other one", "that again".
- If the message asks for a different style, length or level of detail \
("explain simply", "shorten it", "say that again"), the query is the subject \
of the previous answer. Drop the instruction about how to answer.
- Keep the words a document would contain.
- If the message names its own topic and stands alone, return it unchanged.
- Reply with the query alone: one line, no quotes, no preamble, no answer."""


class NvidiaLLMError(Exception):
    """The NVIDIA LLM provider could not complete a request.

    Raised for missing credentials, network failures, bad responses, and
    unparseable replies. It never falls back to guessing.
    """


class NvidiaLLMProvider:
    """Decide one agent-loop action using NVIDIA's chat-completions endpoint.

    Construction always succeeds; :meth:`decide` raises
    :class:`NvidiaLLMError` when credentials or the endpoint are absent, so
    the provider refuses rather than improvises.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        disable_thinking: bool | None = None,
    ) -> None:
        # `.env.example` ships the name with no value, so an empty string is
        # the normal "unset" state here, not a model called "". Using
        # os.environ.get's default would keep the empty value and send
        # {"model": ""} to the endpoint, which fails slowly instead of at once.
        configured_model = os.environ.get("NVIDIA_LLM_MODEL", "").strip()
        self._model_id = model_id or configured_model or DEFAULT_LLM_MODEL
        self._api_key = api_key if api_key is not None else os.environ.get("NVIDIA_API_KEY", "")
        # An explicitly passed empty string means empty, not "fall back to the
        # environment": tests rely on being able to pin a missing setting.
        self._base_url = (
            base_url if base_url is not None else os.environ.get("NVIDIA_API_BASE_URL", "")
        ).rstrip("/")
        self._timeout = timeout
        #: Whether to ask the endpoint to skip the model's reasoning phase.
        #: Only sent when true: a model that does not understand the parameter
        #: is left alone rather than sent an unknown field.
        self._disable_thinking = (
            disable_thinking
            if disable_thinking is not None
            else _env_flag("NVIDIA_LLM_DISABLE_THINKING", DISABLE_THINKING_DEFAULT)
        )

    @property
    def model_id(self) -> str:
        """Identifier recorded in message metadata."""
        return self._model_id

    def decide(
        self,
        *,
        messages: list[dict[str, str]],
        chunks: list[Chunk],
        tools: list[str],
    ) -> LLMAction:
        """One chat completion, parsed into an :class:`LLMAction`.

        ``messages`` is the conversation so far (history plus the current
        question); ``chunks`` are the chunks retrieved for this turn, numbered
        1..N in the order they appear in the prompt. Citations the model
        returns are those numbers, verified again by the agent loop.

        Raises:
            NvidiaLLMError: credentials missing, network failure, HTTP error,
                or a reply that contains no usable JSON object.
        """
        if not self._api_key:
            raise NvidiaLLMError(
                "NVIDIA_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or pass api_key explicitly."
            )
        if not self._base_url:
            raise NvidiaLLMError(
                "NVIDIA_API_BASE_URL is not set. Set it in .env or pass "
                "base_url explicitly."
            )

        payload_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": self._chunks_section(chunks)},
            *messages,
            {
                "role": "system",
                "content": (
                    "Reply now with exactly one JSON object. No prose before "
                    "or after it, no code fences."
                ),
            },
        ]

        content = self._complete(payload_messages)
        parsed = _parse_json_object(content)
        return _action_from_json(parsed)

    def rewrite_follow_up(
        self, *, question: str, history: list[dict[str, str]]
    ) -> str:
        """Rewrite a follow-up into a query retrieval can use.

        The agent loop decides *whether* to call this; this method only
        performs the rewrite. On any failure it raises, and the loop falls
        back to the literal question -- a lost rewrite must never lose the
        question. The result is collapsed to one line because retrieval takes
        a query, not a paragraph.
        """
        transcript = "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in history[-REWRITE_HISTORY_MESSAGES:]
        )
        content = self._complete(
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{transcript}\n\n"
                        f"Follow-up: {question}"
                    ),
                },
            ]
        )
        return _single_line(content)

    def _chunks_section(self, chunks: list[Chunk]) -> str:
        """The numbered chunk listing, or an explicit empty set.

        Saying ``no extracts`` rather than omitting the section keeps the
        model's refusal honest: nothing was shown, so nothing can be cited.
        """
        if not chunks:
            return (
                "Document extracts: none this turn. You may still 'search' "
                "with a rewritten query, or 'refuse'."
            )
        numbered = "\n\n".join(
            f"[{i}] ({chunk.rel_path}, {chunk.location}):\n{chunk.text}"
            for i, chunk in enumerate(chunks, start=1)
        )
        return (
            "Document extracts this turn -- cite these by their numbers only:\n\n"
            f"{numbered}"
        )

    def _complete(self, payload_messages: list[dict[str, str]]) -> str:
        """One blocking chat completion; returns the assistant text."""
        body: dict[str, object] = {
            "model": self._model_id,
            "messages": payload_messages,
            "temperature": 0,
            "max_tokens": MAX_RESPONSE_TOKENS,
        }
        if self._disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        encoded = json.dumps(body).encode("utf-8")

        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise NvidiaLLMError(
                f"NVIDIA chat request failed with HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise NvidiaLLMError(
                f"NVIDIA chat request could not reach the endpoint: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise NvidiaLLMError(
                f"NVIDIA chat request did not finish within {self._timeout:g}s"
            ) from exc

        choices = payload.get("choices", [])
        if not choices:
            raise NvidiaLLMError("NVIDIA returned no choices; cannot decide.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            # Name the shape of the failure: an empty answer after the model
            # spent its budget reasoning is a token-budget problem, not the
            # same thing as an endpoint that returned nothing at all.
            finish_reason = choices[0].get("finish_reason")
            reasoning = message.get("reasoning_content") or ""
            detail = f" (finish_reason={finish_reason}"
            if reasoning:
                detail += f", {len(reasoning)} characters of reasoning"
            detail += ")"
            raise NvidiaLLMError(f"NVIDIA returned an empty completion{detail}.")
        return content


def _single_line(text: str) -> str:
    """Collapse a completion to one whitespace-normalised line.

    Models like to answer with a quoted query on its own line and an
    explanation below it; searching the explanation is not what was asked.
    """
    return " ".join(text.strip().strip('"').split())


def _parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a completion.

    Models are told to answer in bare JSON, but a stray fence or sentence
    must degrade to a clear provider error rather than a fabricated action.
    Braces balance across string literals, so content inside quotes cannot
    confuse the scan.
    """
    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()

    depth = 0
    in_string = False
    escaped = False
    start = -1
    for position, character in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = position
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start != -1:
                blob = candidate[start : position + 1]
                try:
                    parsed = json.loads(blob)
                except json.JSONDecodeError as exc:
                    raise NvidiaLLMError(
                        f"the model's reply contained malformed JSON: {exc}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise NvidiaLLMError(
                        "the model's reply was JSON but not an object."
                    )
                return parsed

    raise NvidiaLLMError(
        "the model's reply contained no JSON object; cannot decide."
    )


#: Aggregate keys each action is allowed to carry, and the target it maps to.
_ACTIONS = {
    "answer": ("text", "citations"),
    "clarification": ("question", "options"),
    "refusal": ("reason",),
    "search": ("query",),
}


def _action_from_json(parsed: dict) -> LLMAction:
    """Turn the parsed JSON object into an :class:`LLMAction`.

    An unrecognised ``action`` becomes ``refusal`` -- never an exception the
    loop would turn into an HTTP 500. The refusal text names what arrived,
    which is a diagnosable failure rather than a manufactured answer.
    """
    action = str(parsed.get("action", "")).strip().lower()

    if action == "answer":
        return LLMAction(
            "answer",
            text=str(parsed.get("text", "") or ""),
            chunk_ids=[int(v) for v in parsed.get("citations", []) or [] if str(v).strip().isdigit()],
        )

    if action == "clarification":
        options = [str(option) for option in parsed.get("options", []) or []]
        return LLMAction(
            "clarification",
            question=str(parsed.get("question", "") or ""),
            options=options,
        )

    if action == "refusal":
        return LLMAction("refusal", reason=str(parsed.get("reason", "") or ""))

    if action == "search":
        return LLMAction("search", query=str(parsed.get("query", "") or ""))

    return LLMAction(
        "refusal",
        reason=(f"the model returned an unrecognized action {action!r}."),
    )
