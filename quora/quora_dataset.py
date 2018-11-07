import torch
import pandas as pd
from typing import Iterator, List, Dict, Optional

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary

@DatasetReader.register("quora_reader")
class QuoraDatasetReader(DatasetReader):
    """
    DatasetReader for Quora questions dataset
    """
    def __init__(self, 
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None, 
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        Read publication and venue dataset in CSV format
        
        """
        reader = pd.read_csv(file_path, chunksize=1)
        target_dict = {0: '0', 1: '1'}
        for row in reader:
            d = dict(row.iloc[0])
            qid = d['qid']
            question_text = d['question_text']
            target = target_dict.get(d['target'], 0)
            yield self.text_to_instance(question_text, target)
        
    def text_to_instance(self, 
                         question_text: str, 
                         target: str=None) -> Instance:
        """
        Turn title, abstract, and venue to instance
        """
        tokenized_question = self._tokenizer.tokenize(question_text)
        question_field = TextField(tokenized_question, self._token_indexers)
        fields = {'question_text': question_field}
        if target is not None:
            fields['target'] = LabelField(target)
        return Instance(fields)