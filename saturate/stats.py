"""Stats: what one run counted. pump() creates it, finishes it and returns it; skip_done() and
drain() add their counts. Its JSON is the run summary on storage and, in agent mode, the one
line on stdout (CONTRACT §5, §7).

It imports nothing from saturate, so every stage can name it in its signature.
"""

from __future__ import annotations

import dataclasses
import json


@dataclasses.dataclass
class Stats:
    rows_total: int = 0
    rows_done_prior: int = 0
    rows_errored_prior: int = 0  # of rows_done_prior: skipped rows whose only record is an error
    rows_processed: int = 0
    rows_failed: int = 0
    rows_deduped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    final_limit: int = 0  # the window at the end (maybe mid-trial); Auto(initial=final_limit) starts near it
    input_bound: bool = False
    breaker_opens: int = 0
    retries: int = 0  # retry attempts over the run (a row that failed three times and then succeeded: 3)
    hints: list[str] = dataclasses.field(default_factory=list)
    # window reductions by controller reason
    cut_reasons: dict[str, int] = dataclasses.field(default_factory=dict)
    # ticks per bottleneck verdict (CONTRACT §6)
    bound_by: dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def rows_unaccounted(self) -> int:
        """Rows read but neither skipped nor handed to the sink; 0 after every run that ends
        normally."""
        accounted = self.rows_done_prior + self.rows_deduped + self.rows_processed + self.rows_failed
        return self.rows_total - accounted

    @property
    def tokens_per_sec(self) -> float:
        total = self.prompt_tokens + self.completion_tokens
        return round(total / self.elapsed_s, 1) if self.elapsed_s else 0.0

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["tokens_per_sec"] = self.tokens_per_sec
        return json.dumps(d)
