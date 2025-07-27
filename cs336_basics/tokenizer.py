import regex as re
import json
from collections import Counter, defaultdict
import heapq
from typing import List, Dict, Tuple, Set, Optional, Iterable, Iterator

class PreTokenizer:
    """
    PreTokenizer负责根据特殊标记和通用文本模式将输入文本分割成初始的字节序列。
    """
    def __init__(self, special_tokens: List[str]):
        self.special_tokens = {token: token.encode('utf-8') for token in special_tokens}
        
        # 用于匹配常规单词的正则表达式
        self.word_pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        
        if self.special_tokens:
            # 按长度对特殊标记进行排序，以便在正则表达式中优先匹配较长的标记
            sorted_special_tokens = sorted(self.special_tokens.keys(), key=len, reverse=True)
            # 用于按特殊标记分割文本的正则表达式
            special_pat_str = "|".join(re.escape(token) for token in sorted_special_tokens)
            self.special_pat = re.compile(f"({special_pat_str})")
        else:
            self.special_pat = None

    def tokenize(self, text: str) -> List[bytes]:
        """
        首先按特殊标记分割文本，然后对非特殊标记的块进行分词。
        """
        if not self.special_pat:
            return [match.group(0).encode('utf-8') for match in self.word_pat.finditer(text)]

        all_tokens = []
        # 使用特殊标记正则表达式来分割文本
        chunks = self.special_pat.split(text)
        
        for chunk in chunks:
            if not chunk:
                continue
            
            # 如果块是特殊标记，直接将其编码后的bytes添加到列表中
            if chunk in self.special_tokens:
                all_tokens.append(self.special_tokens[chunk])
            else:
                # 否则，对这个块应用常规的单词分词规则
                word_tokens = [match.group(0).encode('utf-8') for match in self.word_pat.finditer(chunk)]
                all_tokens.extend(word_tokens)
        return all_tokens

class BPETrainer:
    """
    BPETrainer实现了BPE算法的核心训练逻辑。
    它使用优化的数据结构（如最大堆和反向索引）来高效地找到和合并字节对。
    """
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256.")
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pre_tokenizer = PreTokenizer(special_tokens)
        
        # 初始化词汇表，包括基本字节和特殊标记
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # 为特殊标记分配ID，确保它们不会与基本字节冲突
        for token in self.special_tokens:
            encoded_token = token.encode('utf-8')
            if encoded_token not in self.vocab.values():
                 self.vocab[len(self.vocab)] = encoded_token

    def train(self, text: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        # 1. 预分词并计算词频
        tokens = self.pre_tokenizer.tokenize(text)
        
        # 过滤掉特殊标记，只对常规单词进行BPE训练
        special_tokens_bytes = set(self.pre_tokenizer.special_tokens.values())
        word_freqs = Counter(token for token in tokens if token not in special_tokens_bytes)

        # 2. 初始化数据结构
        splits: Dict[bytes, List[bytes]] = {word: [bytes([b]) for b in word] for word in word_freqs}
        pair_freqs: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
        pair_to_words: Dict[Tuple[bytes, bytes], Set[bytes]] = defaultdict(set)
        freq_max_heap: List[Tuple[int, Tuple[bytes, bytes]]] = []

        # 初始化字节对频率和反向索引
        for word, freq in word_freqs.items():
            word_pieces = splits[word]
            if len(word_pieces) > 1:
                for p1, p2 in zip(word_pieces[:-1], word_pieces[1:]):
                    pair = (p1, p2)
                    pair_freqs[pair] += freq
                    pair_to_words[pair].add(word)

        # 初始化最大堆
        for pair, freq in pair_freqs.items():
            heapq.heappush(freq_max_heap, (-freq, pair))

        # 3. BPE合并循环
        merges: List[Tuple[bytes, bytes]] = []
        num_merges_needed = self.vocab_size - len(self.vocab)
        
        while len(merges) < num_merges_needed:
            best_pair = self._find_best_pair(freq_max_heap, pair_freqs)
            if best_pair is None:
                break

            merges.append(best_pair)
            new_token_bytes = best_pair[0] + best_pair[1]
            self.vocab[len(self.vocab)] = new_token_bytes

            self._update_data_structures(best_pair, new_token_bytes, splits, pair_freqs, pair_to_words, word_freqs, freq_max_heap)

        return self.vocab, merges

    def _find_best_pair(self, freq_max_heap: List, pair_freqs: Dict) -> Optional[Tuple[bytes, bytes]]:
        """
        从最大堆中弹出频率最高的有效字节对，并正确处理平局。
        """
        while freq_max_heap:
            # 1. 弹出频率最高的候选项
            neg_freq, pair = heapq.heappop(freq_max_heap)
            freq = -neg_freq

            # 2. 检查是否是“陈旧”或无效的条目
            if pair not in pair_freqs or pair_freqs[pair] != freq:
                continue

            # 3. 处理平局：查找所有具有相同最高频率的字节对
            best_pair = pair
            candidates = []
            # 查看堆顶是否有更多相同频率的候选项
            while freq_max_heap and freq_max_heap[0][0] == neg_freq:
                _, other_pair = heapq.heappop(freq_max_heap)
                if other_pair in pair_freqs and pair_freqs[other_pair] == freq:
                    # 如果是有效条目，则进行比较
                    if other_pair > best_pair:
                        candidates.append(best_pair)
                        best_pair = other_pair
                    else:
                        candidates.append(other_pair)
            
            # 4. 将未被选中的候选项重新推入堆
            for p in candidates:
                heapq.heappush(freq_max_heap, (neg_freq, p))

            return best_pair
        return None

    def _update_data_structures(self, best_pair: Tuple[bytes, bytes], new_token: bytes, splits: Dict, pair_freqs: Dict, pair_to_words: Dict, word_freqs: Dict, freq_max_heap: List):
        """在合并后，增量更新所有相关的数据结构。"""
        for word in list(pair_to_words.get(best_pair, [])):
            word_freq = word_freqs[word]
            word_pieces = splits[word]
            
            i = 0
            while i < len(word_pieces) - 1:
                if word_pieces[i] == best_pair[0] and word_pieces[i+1] == best_pair[1]:
                    # 合并字节对
                    word_pieces[i] = new_token
                    word_pieces.pop(i+1)

                    # 更新旧的相邻字节对的频率
                    if i > 0:
                        self._update_pair_freq((word_pieces[i-1], best_pair[0]), -word_freq, pair_freqs, pair_to_words, word, freq_max_heap)
                    if i < len(word_pieces) - 1:
                        self._update_pair_freq((best_pair[1], word_pieces[i+1]), -word_freq, pair_freqs, pair_to_words, word, freq_max_heap)

                    # 更新新的相邻字节对的频率
                    if i > 0:
                        self._update_pair_freq((word_pieces[i-1], new_token), word_freq, pair_freqs, pair_to_words, word, freq_max_heap)
                    if i < len(word_pieces) - 1:
                        self._update_pair_freq((new_token, word_pieces[i+1]), word_freq, pair_freqs, pair_to_words, word, freq_max_heap)
                else:
                    i += 1
        
        # 从数据结构中完全移除已合并的字节对
        if best_pair in pair_freqs:
            del pair_freqs[best_pair]
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]

    def _update_pair_freq(self, pair: Tuple[bytes, bytes], freq_delta: int, pair_freqs: Dict, pair_to_words: Dict, word: bytes, freq_max_heap: List):
        """更新单个字节对的频率，并维护反向索引和最大堆。"""
        pair_freqs[pair] += freq_delta
        
        if pair_freqs[pair] > 0:
            pair_to_words[pair].add(word)
            heapq.heappush(freq_max_heap, (-pair_freqs[pair], pair))
        else:
            # 如果频率降至零或以下，则清理
            del pair_freqs[pair]
            if pair in pair_to_words:
                pair_to_words[pair].discard(word)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练BPE分词器的主函数。
    它读取输入文件，实例化BPETrainer，并调用其train方法。
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return {}, []
    
    trainer = BPETrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    vocab, merges = trainer.train(text)
    
    # 对合并规则进行排序，以确保确定性输出
    # 注意：BPE的性质决定了合并顺序就是其优先级
    # merges列表已经是按顺序生成的，所以不需要额外排序
    
    return vocab, merges

# --- 辅助函数（为增量更新而修改） ---

def get_pair_stats(word_counts: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    """统计所有相邻字节对的频率。"""
    pair_stats = defaultdict(int)
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair_stats[(word[i], word[i+1])] += count
    return pair_stats

def merge_pair(word_counts: dict, pair: tuple, new_token_id: int) -> dict:
    """合并所有单词中的指定字节对。"""
    new_word_counts = {}
    for word_bytes, count in word_counts.items():
        if len(word_bytes) == 1:
            new_word_counts[word_bytes] = count
            continue
            
        new_word = []
        i = 0
        while i < len(word_bytes):
            if i < len(word_bytes) - 1 and (word_bytes[i], word_bytes[i+1]) == pair:
                new_word.append(new_token_id)
                i += 2
            else:
                new_word.append(word_bytes[i])
                i += 1
        new_word_counts[tuple(new_word)] = count
    return new_word_counts

class Tokenizer:
    """BPE分词器。"""

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Args:
            vocab: token ID 到 bytes 的词汇表。
            merges: BPE合并规则。
            special_tokens: 特殊字符列表。
        """
        self.vocab = vocab
        self.merges = merges

        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens = {}
        self.special_pattern = None
        if special_tokens:
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)
            for token_str in sorted_tokens:
                self.special_tokens[token_str] = self.byte_to_id[token_str.encode("utf-8")]
            self.special_pattern = "(" + "|".join(re.escape(k) for k in sorted_tokens) + ")"
        self.pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None) -> "Tokenizer":
        """
        从文件构造分词器实例。
        """
        with open(vocab_filepath, 'r') as f:
            # vocab文件通常是json，键是字符串，需要转回整数
            str_keys_vocab = json.load(f)
            vocab = {int(k): v.encode('utf-8') for k, v in str_keys_vocab.items()}

        with open(merges_filepath, 'r', encoding="utf-8") as f:
            merges_str = [line.strip() for line in f]
            # 合并规则文件每行是 "字节1 字节2"，需要解析
            merges = [tuple(part.encode('utf-8') for part in merge.split(' ')) for merge in merges_str]

        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        """
        对不含特殊字符的文本字节块进行BPE编码。
        """
        pre_tokens = [s.encode('utf-8') for s in self.pat.findall(text_bytes.decode('utf-8', errors='replace'))]
        token_ids = []
        for word_bytes in pre_tokens:
            if not word_bytes:
                continue
            parts = [bytes([b]) for b in word_bytes]
            while len(parts) > 1:
                best_pair_info = min(
                    (
                        ((parts[i], parts[i+1]),
                        self.merge_ranks.get((parts[i], parts[i+1]), float('inf')))
                        for i in range(len(parts) - 1)
                    ),
                    key=lambda x: x[1]
                )
                if best_pair_info[1] == float('inf'):
                    break
                best_pair_to_merge = best_pair_info[0]
                new_parts = []
                i = 0
                while i < len(parts):
                    if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair_to_merge:
                        new_parts.append(parts[i] + parts[i+1])
                        i += 2
                    else:
                        new_parts.append(parts[i])
                        i += 1
                parts = new_parts
            for part in parts:
                token_ids.append(self.byte_to_id[part])
        return token_ids

    def encode(self, text: str) -> List[int]:
        """
        编码文本为token ID列表。
        """
        if not self.special_pattern:
            return self._encode_chunk(text.encode('utf-8'))
        chunks = re.split(self.special_pattern, text)
        all_token_ids = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                all_token_ids.append(self.special_tokens[chunk])
            else:
                all_token_ids.extend(self._encode_chunk(chunk.encode('utf-8')))
        return all_token_ids

    def decode(self, ids: List[int]) -> str:
        """
        解码token ID列表为文本。
        """
        all_bytes = b"".join(self.vocab.get(token_id, b'') for token_id in ids)
        text = all_bytes.decode('utf-8', errors='replace')
        return text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        给定字符串可迭代对象，懒惰地生成token ID。
        """
        for text_chunk in iterable:
            encoded_ids = self.encode(text_chunk)
            for token_id in encoded_ids:
                yield token_id