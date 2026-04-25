"""Feature overlays and alternative data transforms."""

from .sentiment import (
    BaseSentimentModel,
    FinBERTSentimentModel,
    NewsSentimentAggregator,
    RuleBasedFinancialSentimentModel,
    SentimentConfig,
    adjust_pair_rankings_with_sentiment,
    apply_sentiment_overlay,
    build_best_available_sentiment_model,
    build_pair_sentiment_overlay,
)

__all__ = [
    "BaseSentimentModel",
    "FinBERTSentimentModel",
    "NewsSentimentAggregator",
    "RuleBasedFinancialSentimentModel",
    "SentimentConfig",
    "adjust_pair_rankings_with_sentiment",
    "apply_sentiment_overlay",
    "build_best_available_sentiment_model",
    "build_pair_sentiment_overlay",
]
