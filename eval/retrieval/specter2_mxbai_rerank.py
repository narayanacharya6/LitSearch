from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import numpy as np
from typing import List, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from eval.retrieval.kv_store import KVStore
from eval.retrieval.kv_store import TextType
from utils import utils
from sentence_transformers import CrossEncoder
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class SPECTER2_MXBAI_RERANK(KVStore):
    def __init__(self, index_name: str, model_path: str = "allenai/specter2::mixedbread-ai/mxbai-rerank-large-v1"):
        super().__init__(index_name, "specter2_mxbai_rerank")
        self.model_path = model_path
        self._init()

    def _init(self):
        self._tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_aug2023refresh_base")
        # self._query_model = AutoModel.from_pretrained("allenai/specter2_base")
        # self._query_model.load_adapter(
        #     "allenai/specter2_adhoc_query",
        #     source="hf",
        #     load_as="specter2_adhoc_query",
        #     set_active=True,
        # )
        # self._query_model = self._query_model.to(DEVICE)

        self._model = AutoAdapterModel.from_pretrained("allenai/specter2_aug2023refresh_base")
        self._model.load_adapter(
            "allenai/specter2_aug2023refresh", source="hf", load_as="specter2_proximity", set_active=True
        )
        self._model = self._model.to(DEVICE)

        self._ce_model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", device=DEVICE)

    def _encode_batch(
        self, texts: List[str], type: TextType, show_progress_bar: bool = True
    ) -> List[Any]:
        model_inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        ).to(DEVICE)

        # if type == TextType.QUERY:
        #     outputs = self._query_model(**model_inputs)
        # elif type == TextType.KEY:
        #     outputs = self._model(**model_inputs)
        # else:
        #     raise RuntimeError()

        outputs = self._model(**model_inputs)
        return outputs.last_hidden_state[:, 0, :].detach().cpu().tolist()

    def _query(self, encoded_query: Any, n: int, query_text: Optional[str] = None) -> List[int]:
        cosine_similarities = cosine_similarity([encoded_query], self.encoded_keys)[0]
        top_indices = cosine_similarities.argsort()[-n:][::-1]

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

    def load(self, path: str):
        super().load(path)
        self._init()
        return self
