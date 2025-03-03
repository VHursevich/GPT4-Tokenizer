import tiktoken
from regex import RegexTokenizer


def bpe(mergeable_ranks, token, max_rank):
  parts = [bytes([b]) for b in token]

  while True:
    min_idx = None
    min_rank = None
    for i, pair in enumerate(zip(parts[:-1], parts[1:])):
      rank = mergeable_ranks.get(pair[0] + pair[1])
      if rank is not None and (min_rank is None or rank < min_rank):
        min_idx = i
        min_rank = rank
    if min_rank is None or (max_rank is not None and min_rank >= max_rank):
      break
    assert min_idx is not None
    parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:] 

  return parts


def recover_merges(mergeable_ranks):
  merges = {}

  for token, rank in mergeable_ranks:
    if len(token) == 1:
      continue
    pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
    assert len(pair) == 2

    ix0 = mergeable_ranks[pair[0]]
    ix1 = mergeable_ranks[pair[1]]
    merges[(ix0, ix1)] = rank
  
  return merges


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
  def __init__(self):
    super().__init__(pattern=GPT4_SPLIT_PATTERN)
    # get the official tokenizer 
    enc = tiktoken.get_encoding("cl100k_base")
    mergeable_ranks = enc.__mergeable_ranks

    self.merges = recover_merges(mergeable_ranks)

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (pc0, pc1), idx in self.merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    
    self.vocab = vocab

    self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
    self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

    self.register_special_tokens(GPT4_SPECIAL_TOKENS)





