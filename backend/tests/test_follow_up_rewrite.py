"""The follow-up rules: which messages are rewritten, and which restyled.

These are the pure decisions the agent loop makes before it spends a model
call or builds a prompt. They are tested away from the loop because each one
is a judgement about language, and a failing language judgement is easier to
read on its own than inside a six-message prompt.
"""

from __future__ import annotations

from verity.agent.rewrite import looks_like_follow_up, looks_like_restyle

HISTORY = [
    {"role": "user", "content": "What is the maximum junction temperature?"},
    {"role": "assistant", "content": "94 C."},
]


def test_a_short_message_after_a_turn_is_a_follow_up() -> None:
    assert looks_like_follow_up("explain in simple manner", HISTORY)


def test_a_pronoun_makes_even_a_long_message_a_follow_up() -> None:
    assert looks_like_follow_up(
        "Could you tell me whether that limit applies when the board is upside down?",
        HISTORY,
    )


def test_a_long_self_contained_question_is_not_a_follow_up() -> None:
    assert not looks_like_follow_up(
        "What is the maximum junction temperature of the reactor core?",
        HISTORY,
    )


def test_the_first_question_is_never_a_follow_up() -> None:
    assert not looks_like_follow_up("explain in simple manner", [])


def test_asks_to_simplify_is_a_restyle() -> None:
    assert looks_like_restyle("explain in simple manner")
    assert looks_like_restyle("ELI5")
    assert looks_like_restyle("can you rephrase that?")


def test_saying_it_was_not_understood_is_a_restyle() -> None:
    """The same request in the user's own words, not a new question."""
    assert looks_like_restyle("i am not understanding")
    assert looks_like_restyle("I don't understand")
    assert looks_like_restyle("I don’t understand")
    assert looks_like_restyle("that makes no sense")
    assert looks_like_restyle("sorry, you lost me")


def test_a_not_understood_message_is_also_a_follow_up() -> None:
    """It must be rewritten to the previous topic before it is searched."""
    assert looks_like_follow_up("i am not understanding", HISTORY)


def test_in_other_words_is_a_restyle() -> None:
    assert looks_like_restyle("in other words?")
    assert looks_like_restyle("say that again in plain english")


def test_a_new_task_is_not_a_restyle() -> None:
    """Summarising the contract is different work from restating an answer."""
    assert not looks_like_restyle("summarise the contract")
    assert not looks_like_restyle("what about the warranty?")
    assert not looks_like_restyle("and what about it in celsius?")
