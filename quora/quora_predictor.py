from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor

@Predictor.register('quora_predictor')
class QuoraQuestionPredictor(Predictor):
    """"
    Predictor for Quora Questions
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        question_text = json_dict['question_text']
        instance = self._dataset_reader.text_to_instance(question_text=question_text)
        return instance