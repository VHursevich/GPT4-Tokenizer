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
    ix1 = mergeable_ranks[pair[1]]Ницан Тор
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
    for (p0, p1), idx in self.merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    
    self.vocab = vocab

    self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
    self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

    self.register_special_tokens(GPT4_SPECIAL_TOKENS)

  def _encode_chunk(self, text_bytes):
    text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
    ids = super()._encode_chunk(text_bytes)
    return ids

  def decode(self, ids):
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
    text = text_bytes.decode("utf-8", errors="replace")
    
    return text
  
  def train(self, text, vocab_size, verbose=False):
    raise NotImplementedError("GPT4Tokenizer cannot be trained")
    
  def load(self, model_file):
    raise NotImplementedError("GPT4Tokenizer cannot be loaded")

  def save(self, file_prefix):
    raise NotImplementedError("GPT4Tokenizer cannot be saved")
  
  def save_vocab(self, vocab_file):
    from base import render_token

    vocab = {idx: bytes([self.inverse_byte_shuffle[idx]] for idx in range(256))}
    for (p0, p1), idx in self.merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]

    inverted_merges = {idx: pair for pair, idx in self.merges.items()}

    with open(vocab_file, "w", encoding="utf-8") as f:
      for idx, token in vocab.items():
        s = render_token(token)
        if idx in inverted_merges:
          idx0, idx1 = inverted_merges[idx]
          s0 = render_token(vocab[idx0])
          s1 = render_token(vocab[idx1])
          f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
        else:
          f.write(f"[{s}] {idx}\n")
          
