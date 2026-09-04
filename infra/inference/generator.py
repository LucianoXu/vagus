# Stateful autoregressive generation.
#
# The generator holds one continuous token stream per batch row and
# exposes it as three verbs: reset, prefill (push tokens in), gen (pull
# tokens out). Stateless "prompt -> completion" is the composition
# reset + prefill + gen, provided as a convenience, not as the primitive.
#
# State invariant:  model state  +  pending  ==  the whole stream.
#   `pending` is the newest token of the stream, (B,) int64, which the
#   model has NOT consumed yet. prefill feeds everything but the last new
#   token and makes that token pending; a gen step feeds the pending token
#   (L == 1), samples from the resulting logits, and the sample becomes
#   pending. So every token is fed exactly once, logits are computed
#   exactly once per sampled position (prefill passes return_logits=False
#   and never materialises them; the deferred last token is what makes
#   that possible), the state is closed after every step, and a stream
#   consumer may break at any point without corrupting it.
#
# Batch: the tensor interface is (B, L) throughout, but v1 supports only
# equal-length prompts — decode_step has no attention mask, so ragged
# rows cannot be left-padded. Rows stop together: the stream ends when
# every row has emitted a stop id or max_new_tokens is reached.

from typing import TYPE_CHECKING, Iterator, Sequence

import torch

from ..models.decodable import Decodable, missing_decodable
from ..utils import infer_device
from .sampling import SamplingConfig, sample

if TYPE_CHECKING:
    from tokenizers import Tokenizer

class Generator:
    '''Generation over any model implementing the Decodable protocol.
    Token-level primitives (reset / prefill_ids / gen_ids_stream /
    export_state / load_state) plus text-level conveniences built on an
    optional HF `tokenizers.Tokenizer`. Other model families (HF baselines)
    plug in through a Decodable adapter, not a Generator subclass.

    Stream conventions are the caller's to supply — they belong to how
    the training stream was built (tokenizer META + packing), not to the
    architecture, and the Generator assumes nothing about the vocabulary:
      start_id: how a document starts; reset() makes it the pending
        token, so a fresh stream is never empty and gen() without a
        prompt is unconditional generation. For infra/tokenizers/mistral32k
        this is BOS.
      stop_ids: how a document ends; the default for
        SamplingConfig.stop_ids=None. For mistral32k also (BOS,) — packed
        streams separate documents with BOS and never contain EOS.
    start_id None and stop_ids empty -> raw stream, run to max_new_tokens.'''

    def __init__(self, model: Decodable, tokenizer: 'Tokenizer | None' = None,
                 start_id: int | None = None, stop_ids: tuple[int, ...] = ()):
        missing = missing_decodable(model)
        assert not missing, f'{type(model).__name__} lacks Decodable members {missing}'
        self.model: Decodable = model
        self.tokenizer = tokenizer           # None -> ids-only interface
        self.start_id = start_id
        self.stop_ids = tuple(stop_ids)
        self.device = infer_device(model)
        self.rng = torch.Generator(device=self.device)
        self.rng.seed()
        self.batch_size = 0
        self.max_len = 0
        self.fed_len = 0                     # tokens the model has consumed
        self.pending: torch.Tensor | None = None
        self.stop_reason: str | None = None  # 'stop_id' | 'max_new_tokens' | 'capacity'

    # --- state -------------------------------------------------------

    def _resolve_max_len(self, max_len: int | None) -> int:
        if max_len is None:
            max_len = self.model.max_stream_len
        if max_len is None:
            raise ValueError('model has no max_stream_len; pass max_len to reset()')
        return int(max_len)

    def reset(self, batch_size: int = 1, max_len: int | None = None) -> None:
        '''Start `batch_size` streams able to hold max_len tokens, each
        holding just start_id (or nothing).'''
        self.batch_size = batch_size
        self.max_len = self._resolve_max_len(max_len)
        self.model.reset_cache(batch_size, self.max_len)
        self.fed_len = 0
        self.pending = None if self.start_id is None else torch.full(
            (batch_size,), self.start_id, dtype=torch.int64, device=self.device)
        self.stop_reason = None

    @property
    def stream_len(self) -> int:
        '''Tokens in each stream, pending included.'''
        return self.fed_len + (0 if self.pending is None else 1)

    def export_state(self) -> dict:
        '''A self-contained snapshot (model state, pending, RNG).'''
        return {
            'cache': self.model.export_cache(),
            'pending': None if self.pending is None else self.pending.clone(),
            'fed_len': self.fed_len,
            'batch_size': self.batch_size,
            'max_len': self.max_len,
            'rng': self.rng.get_state(),
        }

    def load_state(self, state: dict, max_len: int | None = None) -> None:
        self.batch_size = state['batch_size']
        self.max_len = self._resolve_max_len(max_len or state['max_len'])
        self.model.load_cache(state['cache'], max_cache_len=self.max_len)
        self.pending = None if state['pending'] is None else state['pending'].to(self.device)
        self.fed_len = state['fed_len']
        self.rng.set_state(state['rng'])
        self.stop_reason = None

    # --- feeding -----------------------------------------------------

    @torch.no_grad()
    def _feed(self, block: torch.Tensor, want_logits: bool) -> torch.Tensor | None:
        '''Push a (B, L) block through the model. Returns the last
        position's logits (B, vocab) when wanted, else None.'''
        assert self.batch_size > 0, 'call reset() first'
        assert block.shape[0] == self.batch_size
        L = block.shape[1]
        if self.fed_len + L > self.max_len:
            raise RuntimeError(
                f'stream capacity {self.max_len} exceeded: {self.fed_len} fed + {L} new')
        if L == 0:
            return None
        out = self.model.decode_step(block.to(self.device), return_logits=want_logits)
        self.fed_len += L
        if not want_logits:
            return None
        assert out is not None
        return out[:, -1]

    def prefill_ids(self, ids: torch.Tensor) -> None:
        '''Append ids (B, L) int64, L >= 1, to the streams.'''
        assert ids.dim() == 2 and ids.shape[1] >= 1, 'ids must be (B, L) with L >= 1'
        ids = ids.to(self.device, dtype=torch.int64)
        block = ids[:, :-1] if self.pending is None else torch.cat([self.pending[:, None], ids[:, :-1]], dim=1)
        self._feed(block, want_logits=False)
        self.pending = ids[:, -1].clone()

    # --- generation --------------------------------------------------

    @torch.no_grad()
    def gen_ids_stream(self, cfg: SamplingConfig) -> Iterator[torch.Tensor]:
        '''Yield one (B,) int64 tensor per generated position; each is
        already part of the state when yielded. Stop ids are yielded too;
        the iterator ending is the stop signal (see stop_reason).'''
        assert self.pending is not None, 'empty stream'
        if cfg.seed is not None:
            self.rng.manual_seed(cfg.seed)
        stop_ids = self.stop_ids if cfg.stop_ids is None else cfg.stop_ids
        stop_t = torch.tensor(list(stop_ids), dtype=torch.int64, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.stop_reason = None

        for _ in range(cfg.max_new_tokens):
            if self.fed_len >= self.max_len:
                self.stop_reason = 'capacity'
                return
            logits = self._feed(self.pending[:, None], want_logits=True)   # (B, V)
            assert logits is not None
            t = sample(logits, cfg, self.rng)
            self.pending = t                            # state closed before yield
            if stop_t.numel():
                done |= torch.isin(t, stop_t)
            yield t
            if bool(done.all()):
                self.stop_reason = 'stop_id'
                return
        self.stop_reason = 'max_new_tokens'

    def gen_ids(self, cfg: SamplingConfig) -> torch.Tensor:
        '''(B, N) int64, N <= cfg.max_new_tokens.'''
        steps = list(self.gen_ids_stream(cfg))
        if not steps:
            return torch.zeros(self.batch_size, 0, dtype=torch.int64, device=self.device)
        return torch.stack(steps, dim=1)

    def generate_ids(self, ids: torch.Tensor, cfg: SamplingConfig,
                     max_len: int | None = None) -> torch.Tensor:
        '''Stateless: fresh streams, prompt in, completion out.'''
        self.reset(ids.shape[0], max_len)
        self.prefill_ids(ids)
        return self.gen_ids(cfg)

    # --- text level --------------------------------------------------

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        '''(B, L) int64, raw pieces: no special tokens — document
        boundaries are start_id / stop_ids, handled by the stream.'''
        assert self.tokenizer is not None, 'no tokenizer: use the *_ids methods'
        encs = self.tokenizer.encode_batch(list(texts), add_special_tokens=False)
        lens = {len(e.ids) for e in encs}
        if len(lens) != 1:
            raise ValueError(f'v1 batches need equal-length prompts, got lengths {sorted(lens)}')
        return torch.tensor([e.ids for e in encs], dtype=torch.int64, device=self.device)

    def decode(self, ids: torch.Tensor) -> list[str]:
        assert self.tokenizer is not None
        return self.tokenizer.decode_batch(ids.tolist(), skip_special_tokens=True)

    def prefill(self, texts: Sequence[str] | str) -> None:
        if isinstance(texts, str):
            texts = [texts]
        self.prefill_ids(self.encode(texts))

    def gen_stream(self, cfg: SamplingConfig) -> Iterator[list[str]]:
        '''Yield per step the text delta of each row. Token pieces are not
        prefix-stable on their own (sentencepiece word markers, multi-byte
        characters split across tokens), so each step decodes the whole
        new segment and emits what grew, holding back a trailing
        replacement character until the sequence completes.'''
        B = self.batch_size
        ids: list[list[int]] = [[] for _ in range(B)]
        emitted = [''] * B
        for step in self.gen_ids_stream(cfg):
            for b, t in enumerate(step.tolist()):
                ids[b].append(t)
            texts = self.decode(torch.tensor(ids, dtype=torch.int64))
            delta = []
            for b in range(B):
                full = texts[b]
                if full.endswith('�'):
                    full = full.rstrip('�')
                if full.startswith(emitted[b]):
                    delta.append(full[len(emitted[b]):])
                    emitted[b] = full
                else:   # decoder revised earlier text (rare); resend it all
                    delta.append(full)
                    emitted[b] = full
            yield delta
        # flush anything held back
        if ids and ids[0]:
            texts = self.decode(torch.tensor(ids, dtype=torch.int64))
            tail = [t[len(e):] if t.startswith(e) else t for t, e in zip(texts, emitted)]
            if any(tail):
                yield tail

    def gen(self, cfg: SamplingConfig) -> list[str]:
        return self.decode(self.gen_ids(cfg))

    def generate(self, texts: Sequence[str] | str, cfg: SamplingConfig,
                 max_len: int | None = None) -> list[str]:
        '''Stateless text convenience: reset + prefill + gen.'''
        if isinstance(texts, str):
            texts = [texts]
        self.reset(len(texts), max_len)
        self.prefill(texts)
        return self.gen(cfg)
