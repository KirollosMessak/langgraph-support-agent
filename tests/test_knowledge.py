from support_agent.knowledge import PolicySearch, load_sections


def test_sections_are_split_by_heading():
    titles = [s.title for s in load_sections()]
    assert "Returns and refunds > Return window" in titles
    assert len(titles) >= 12


def test_search_finds_the_right_section():
    search = PolicySearch()
    assert "Shipping fees" in search.search("free shipping cost minimum")[0].title
    assert "Warranty" in search.search("warranty electronics months")[0].title
    assert "When you can cancel" in search.search("cancel order before shipping")[0].title


def test_search_matches_singular_and_plural_and_finds_delivery_area():
    search = PolicySearch()
    assert "Delivery times" in search.search("delivery time Alexandria")[0].title
    assert "Where we deliver" in search.search("international shipping Saudi Arabia")[0].title
