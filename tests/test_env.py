import math

from learner.env import Binary, State, Unary, shaped_reward


def test_solved_within_epsilon():
    state = State.from_values([24.0, 1.0], 24.0)
    assert state.solved()
    assert not State.from_values([24.001], 24.0).solved()


def test_depth_cap_blocks_fourth_unary():
    state = State.from_values([16.0], 2.0)
    s1 = state.step(Unary(i=0, op="sqrt"))
    s2 = s1.step(Unary(i=0, op="sqrt"))
    assert s2.elems[0].value == 2.0
    s3 = s2.step(Unary(i=0, op="sqrt"))
    assert math.isclose(s3.elems[0].value, math.sqrt(2.0))
    assert s3.elems[0].depth == 3
    assert s3.step(Unary(i=0, op="sqrt")) is None


def test_magnitude_and_domain_guards():
    boom = State.from_values([1e7, 1e7], 1.0).step(Binary(i=0, j=1, op="mul"))
    neg_sqrt = State.from_values([-4.0], 2.0).step(Unary(i=0, op="sqrt"))
    ln_zero = State.from_values([0.0], 2.0).step(Unary(i=0, op="ln"))
    div_zero = State.from_values([1.0, 0.0], 2.0).step(Binary(i=0, j=1, op="div"))
    assert boom is None
    assert neg_sqrt is None
    assert ln_zero is None
    assert div_zero is None


def test_binary_reset_and_unary_advance_depth():
    state = State.from_values([16.0, 2.0], 18.0)
    once = state.step(Unary(i=0, op="sqrt"))
    assert once.elems[0].value == 4.0
    deepened = once.step(Unary(i=0, op="sqrt"))
    assert deepened.elems[0].depth == 2
    assert deepened.elems[0].value == 2.0
    merged = deepened.step(Binary(i=0, j=1, op="add"))
    assert merged.elems[0].depth == 0
    assert merged.elems[0].value == 4.0


def test_shaped_reward_prefers_closer_states():
    near = State.from_values([23.999], 24.0)
    far = State.from_values([-100.0], 24.0)
    assert shaped_reward(near) > shaped_reward(far)


def test_action_counts_match_formula():
    n = 5
    state = State.from_values([1.0] * n, 24.0)
    pairs = n * (n - 1) // 2
    expected_unary = n * 3
    expected_binary = pairs * (2 + 2 * 3)
    actions = state.legal_actions()
    assert len(actions) == expected_unary + expected_binary
