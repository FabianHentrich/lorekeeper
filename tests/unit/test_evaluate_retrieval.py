from evaluation.evaluate_retrieval import section_hit


class TestSectionHit:
    """Matching retrieved chunk headings against a numbered rulebook section."""

    def test_exact_section_matches(self):
        """The section's own heading counts as a hit."""
        assert section_hit("4.10", ["**4.10 Kampfunfähigkeit und Tod** > **Kampfunfähigkeit**"])

    def test_subsection_matches_parent(self):
        """A sub-section lies inside its parent section."""
        assert section_hit("5.4", ["**5.4.1 Krieger** > **Wuchtschlag**"])

    def test_prefix_of_other_number_does_not_match(self):
        """'4.1' must not match '4.10' and vice versa."""
        assert not section_hit("4.1", ["**4.10 Kampfunfähigkeit und Tod**"])
        assert not section_hit("4.10", ["**4.1 Der Ablauf eines Kampfes**"])

    def test_neighbouring_section_misses(self):
        """The Überschaden/Überheilung confusion is a section miss."""
        assert not section_hit("4.10", ["**4.11 Kampfende** > **Überheilung**", "Orte > Arkenfeld"])

    def test_empty_headings(self):
        """No chunks → no hit."""
        assert not section_hit("2.5", [])
