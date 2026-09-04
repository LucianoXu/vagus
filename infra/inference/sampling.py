# Sampling configuration and the single logits -> token step.

from dataclasses import dataclass, asdict

import torch


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0      # 0 -> greedy (top_k / top_p ignored)
    top_k: int | None = None      # keep the k most likely tokens
    top_p: float | None = None    # nucleus: smallest set with mass >= top_p
    # tokens that end the generation (they are still emitted and fed, so
    # the state stays consistent). None -> the Generator's own stop_ids;
    # () -> none, run to max_new_tokens.
    stop_ids: tuple[int, ...] | None = None
    seed: int | None = None       # reseed the generator's RNG at gen start

    def __post_init__(self):
        assert self.max_new_tokens >= 0
        assert self.temperature >= 0
        assert self.top_k is None or self.top_k >= 1
        assert self.top_p is None or 0 < self.top_p <= 1

    def asdict(self) -> dict:
        return asdict(self)


def sample(logits: torch.Tensor, cfg: SamplingConfig,
           rng: torch.Generator | None = None) -> torch.Tensor:
    '''logits (B, V) -> token ids (B,) int64. Computed in fp32.'''
    logits = logits.float()
    if cfg.temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / cfg.temperature

    if cfg.top_k is not None and cfg.top_k < logits.shape[-1]:
        kth = torch.topk(logits, cfg.top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float('-inf'))

    if cfg.top_p is not None and cfg.top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # drop tokens whose *preceding* cumulative mass already reached
        # top_p; the most likely token always survives
        drop = (cum - torch.softmax(sorted_logits, dim=-1)) >= cfg.top_p
        sorted_logits = sorted_logits.masked_fill(drop, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1)
