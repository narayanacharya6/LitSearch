import nltk
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Any, Optional
from eval.retrieval.kv_store import KVStore
from eval.retrieval.kv_store import TextType
from utils import utils
from sentence_transformers import CrossEncoder
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class BM25_MXBAI_RERANK(KVStore):
    def __init__(self, index_name: str):
        super().__init__(index_name, 'bm25_mxbai_rerank')

        nltk.download('punkt')
        nltk.download('stopwords')

        self._tokenizer = nltk.word_tokenize
        self._stop_words = set(nltk.corpus.stopwords.words('english'))
        self._stemmer = nltk.stem.PorterStemmer().stem
        self.index = None   # BM25 index

        self._ce_model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", device=DEVICE)

    def _encode_batch(self, texts: str, type: TextType, show_progress_bar: bool = True) -> List[str]:
        # lowercase, tokenize, remove stopwords, and stem
        tokens_list = []
        for text in texts:
            tokens = self._tokenizer(text.lower())
            tokens = [token for token in tokens if token not in self._stop_words]
            tokens = [self._stemmer(token) for token in tokens]
            tokens_list.append(tokens)
        return tokens_list

    def _query(self, encoded_query: List[str], n: int, query_text: Optional[str] = None) -> List[int]:
        top_indices = np.argsort(self.index.get_scores(encoded_query))[::-1][:n].tolist()
        
        all_scores = []
        batch_size = n
        for batch_index, top_indices_batch in enumerate(utils.chunked(top_indices, batch_size)):
            keys = [self.keys[i] for i in top_indices_batch]
            results = self._ce_model.rank(query_text, keys)
            scores = [doc['score'] for doc in sorted(results, key=lambda x:x['corpus_id'])]
            all_scores.extend(scores)

        assert len(all_scores) == len(top_indices)
        sorted_top_indices = [x for _, x in sorted(zip(all_scores, top_indices), reverse=True)]
        return sorted_top_indices

    def clear(self) -> None:
        super().clear()
        self.index = None
    
    def create_index(self, key_value_pairs: List[Tuple[str, Any]]) -> None:
        super().create_index(key_value_pairs)
        self.index = BM25Okapi(self.encoded_keys)
    
    def load(self, dir_name: str) -> None:
        super().load(dir_name)
        self._tokenizer = nltk.word_tokenize
        self._stop_words = set(nltk.corpus.stopwords.words('english'))
        self._stemmer = nltk.stem.PorterStemmer().stem
        self._ce_model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", device=DEVICE)
        return self