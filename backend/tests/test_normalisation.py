"""Tests for the single normalisation used everywhere.

The contract grades an answer by value, not spelling, so this function decides
whether an extracted value counts as the answer. A bug here does not crash
anything -- it silently makes correct answers look absent, or makes wrong ones
look present -- so every transformation the contract names is pinned here
individually rather than through one end-to-end example.
"""

from __future__ import annotations

import pytest
from verity.engine.normalisation import contains, normalise


class TestTheContractRules:
    """Each clause of the contract's normalisation, tested on its own."""

    def test_case_is_folded(self) -> None:
        assert normalise("rev c2") == normalise("REV-C2")

    def test_hyphens_are_deleted(self) -> None:
        assert normalise("ORR-FAN-2214-B") == "ORRFAN2214B"

    def test_dots_are_deleted(self) -> None:
        """The reason ``4.3.2`` and ``432`` are the same value."""
        assert normalise("4.3.2") == "432"

    def test_underscores_are_deleted(self) -> None:
        assert normalise("REV_C2") == "REVC2"

    def test_middle_dots_are_deleted(self) -> None:
        assert normalise("REV·C2") == "REVC2"

    def test_whitespace_is_deleted(self) -> None:
        assert normalise("Q3 FY27") == "Q3FY27"

    def test_degree_signs_are_deleted(self) -> None:
        assert normalise("94 °C") == "94C"

    def test_degree_signs_do_not_become_letters(self) -> None:
        """``º`` is visually a degree sign but is a distinct Unicode character.

        NFKC folds it to the letter ``o``, which would leave that letter sitting
        inside a value. It has to be removed before folding, not after.
        """
        assert normalise("20º") == "20"

    def test_units_need_no_special_handling(self) -> None:
        """The contract says *ignore* units, and substring matching does that.

        An expected answer of ``94`` must be found in a line that spells it
        ``94 °C``. Because the line normalises to a single token containing
        ``94``, the unit is simply not part of the comparison -- there is no
        unit list to maintain and nothing to strip.
        """
        line = "Maximum junction temperature: 94 °C"

        assert contains(line, "94")

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("94", "94"),
            ("-", ""),
            ("", ""),
            ("...", ""),
            ("  spaced  out  ", "SPACEDOUT"),
        ],
    )
    def test_degenerate_inputs(self, value: str, expected: str) -> None:
        assert normalise(value) == expected


class TestUnicodeFolding:
    """PDF extraction emits spellings a keyboard cannot produce."""

    def test_full_width_digits_fold(self) -> None:
        assert normalise("９４") == "94"

    def test_full_width_separators_fold_then_disappear(self) -> None:
        """The reason separators are deleted *after* NFKC, not before.

        ``４．３`` uses a full-width full stop. Deleting ASCII dots first would
        leave that character behind, and the value would not match ``4.3``.
        """
        assert normalise("４．３") == "43"

    def test_ligatures_fold(self) -> None:
        assert normalise("speciﬁcation") == "SPECIFICATION"

    def test_non_breaking_spaces_are_removed(self) -> None:
        """A non-breaking space is not the same character as a space.

        A naive ``replace(" ", "")`` would leave it in place and split a value
        that the corpus spells as one token.
        """
        assert normalise("Q3 FY27") == "Q3FY27"

    def test_thin_spaces_are_removed(self) -> None:
        assert normalise("1 234") == "1234"


class TestContains:
    """The comparison used for every answer check in the project."""

    def test_matches_across_spelling_differences(self) -> None:
        assert contains("part ORR FAN 2214 B fitted", "ORR-FAN-2214-B")

    def test_matches_across_case(self) -> None:
        assert contains("the revision is rev-c2", "REV-C2")

    def test_does_not_match_a_different_value(self) -> None:
        """The near-miss: 105 is not 94, and must not be read as it."""
        assert not contains("Maximum junction temperature: 105 °C", "94")

    def test_an_empty_needle_never_matches(self) -> None:
        """Otherwise every file would "contain" an empty answer.

        An empty expected answer means the question is a refusal, and a
        comparison that reported a match for it would turn a correct refusal
        into a wrong citation.
        """
        assert not contains("anything at all", "")

    def test_a_punctuation_only_needle_never_matches(self) -> None:
        """Normalisation can reduce a needle to nothing; that is not a match."""
        assert not contains("anything at all", "---")

    def test_is_idempotent(self) -> None:
        """Applying it twice must change nothing, or it is not a canonical form."""
        for value in ("94 °C", "ORR-FAN-2214-B", " 4.3.2 ", "ＲＥＶ Ｃ２"):
            once = normalise(value)
            assert normalise(once) == once
