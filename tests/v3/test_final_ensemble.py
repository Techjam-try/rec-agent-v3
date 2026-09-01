import numpy as np

from research_agent_v3.final_ensemble import percentile_by_user


def test_percentile_by_user_ranks_only_within_each_user():
    users = ["a", "b", "a", "b", "a"]
    scores = np.asarray([3.0, 100.0, 1.0, 50.0, 2.0])

    ranked = percentile_by_user(users, scores)

    np.testing.assert_allclose(ranked, [1.0, 1.0, 0.0, 0.0, 0.5])
