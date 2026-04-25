"""Research helpers for universe screening and candidate selection."""

from .screening import PairScreenConfig, find_candidate_pairs, generate_sector_pairs, rank_sector_pairs, score_candidate_pair

__all__ = [
    "PairScreenConfig",
    "find_candidate_pairs",
    "generate_sector_pairs",
    "rank_sector_pairs",
    "score_candidate_pair",
]
