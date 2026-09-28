import torch
import pytest
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.utils.speech_quality.sage_model.sage_api import SageAPI
from llm_val.val_test.model_quality.valtest_speech_quality import valtest_speech_quality

class LiteracyModelHugWrapper:
    def __init__(self,
                 model_name: str,
                 device: str='cpu'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device
     
    def generate(self,
                 batch,
                 config):
        with torch.no_grad():
            return self.model(torch.tensor(batch['input_ids'], device=self.device),
                            torch.tensor(batch['attention_mask'], device=self.device),
                            **config)
        
    def to_device(self):
        self.model.to(self.device)
        self.model.eval()


class LiteracyTokenizerHugWrapper:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                                                                                                                          
    def encode(self, batch, config):
        return self.tokenizer(batch, **config)


@pytest.fixture()
def create_sampler():
    data_la = joblib.load('data/data_longanswer.pkl')
    sampler_la = LongAnswerSampler(data_la)
    literacy_model_name = 'RussianNLP/ruRoBERTa-large-rucola'
    literacy_model = LiteracyModelHugWrapper(literacy_model_name, device='cuda')
    literacy_tokenizer = LiteracyTokenizerHugWrapper(literacy_model_name)
    literacy_model_config = {}
    literacy_tokenizer_config = {'encode_config': {'max_length': 512,
                                                'padding': 'max_length',
                                                'truncation': True,
                                                'return_token_type_ids': False},
                                'decode_config': {'skip_special_tokens': True}}
    sage_model_api = SageAPI(contour='mlspace')
    res = valtest_speech_quality(sampler=sampler_la,
                                 literacy_model=literacy_model,
                                 literacy_tokenizer=literacy_tokenizer,
                                 literacy_model_config=literacy_model_config,
                                 literacy_tokenizer_config=literacy_tokenizer_config,
                                 test_spelling=True,
                                 spelling_model=sage_model_api)
    return res


def test_valtest_speech_quality_dict(create_sampler):
    test_res = create_sampler
    assert isinstance(test_res, dict)


def test_valtest_speech_quality_keys(create_sampler):
    test_res = create_sampler
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_speech_quality_report(create_sampler):
    test_res = create_sampler
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert set(report['result_dataframes'][0]['Метрика'].values) == set(['Linguistic Acceptability score', 'COLE', 'WER'])


def test_valtest_speech_quality_precomputed(create_sampler):
    test_res = create_sampler
    precomputed = test_res['precomputed']
    assert set(precomputed['metrics_results'].keys()) == set(['literacy_score', 'readability_score', 'spelling_score'])
    assert isinstance(precomputed, dict)