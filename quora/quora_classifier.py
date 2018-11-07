from typing import Iterator, List, Dict, Optional
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules import FeedForward
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.vocabulary import Vocabulary


@Model.register("quora_classifier")
class QuoraQuestionClassifier(Model):
    """
    Quora question classifier model
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 question_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(QuoraQuestionClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.question_encoder = question_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                question_text: Dict[str, torch.LongTensor],
                target: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_question = self.text_field_embedder(question_text)
        question_mask = get_text_field_mask(question_text)
        encoded_question = self.question_encoder(embedded_question, question_mask)

        logits = self.classifier_feedforward(encoded_question)
        class_probabilities = F.softmax(logits, dim=-1)
        argmax_indices = np.argmax(class_probabilities.cpu().data.numpy(), axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace='labels')
                  for x in argmax_indices]
        
        output_dict = {
            'logits': logits, 
            'class_probabilities': class_probabilities, 
            'label': labels
        }
        if target is not None:
            loss = self.loss(logits, target.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, target.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}