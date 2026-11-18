from experiments.standalone.accept import accepted_prefix, summarise, tau_table


def test_prefix_stops_at_first_disagreement():
    assert accepted_prefix([True, True, False, True]) == 2
    assert accepted_prefix([True] * 5) == 5
    assert accepted_prefix([False] + [True] * 5) == 0


def test_tau_uses_min_of_prefix_and_width():
    rows = [{"block": 16, "accepted": 12}, {"block": 16, "accepted": 2}]
    table = tau_table(rows, widths=[4, 9])
    assert table[(16, 4)] == (4 + 2) / 2
    assert table[(16, 9)] == (9 + 2) / 2


def test_width_above_block_is_absent_not_clamped():
    rows = [{"block": 9, "accepted": 5}]
    assert (9, 12) not in tau_table(rows, widths=[12])


def test_summarise_win_and_lose():
    win = [{"block": 16, "accepted": 12}, {"block": 9, "accepted": 5}]
    assert "WIN" in summarise(win, k=9)
    lose = [{"block": 16, "accepted": 5}, {"block": 9, "accepted": 5}]
    assert "LOSE" in summarise(lose, k=9)
