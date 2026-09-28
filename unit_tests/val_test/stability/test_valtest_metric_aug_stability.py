import torch
import pytest
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from llm_val.utils.metrics import METRICS
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel
from llm_val.val_test.stability.valtest_metric_aug_stability import valtest_metric_aug_stability

DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_RUBQ = 'data/data_rubq.pkl'

PARAMS = [(RuBQSampler, DATA_PATH_RUBQ, MultiThreadGigaModel, {'n_jobs': 8}),
          (MMLUSampler, DATA_PATH_MMLU, MultiThreadGigaModel, {'n_jobs': 8})]


class ParaphraseModelHugWrapper:
    def __init__(self,
                 model_name: str,
                 device: str='cpu'):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device
     
    def generate(self,
                 batch,
                 config):
        with torch.no_grad():
            return self.model.generate(**batch, **config)
     
    def to_device(self):
        self.model.to(self.device)
        self.model.eval()


class ParaphraseTokenizerHugWrapper:
    def __init__(self,
                 model_name: str,
                 device: str='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.device = device
     
    def decode(self,
               batch,
               config):
        return self.tokenizer.decode(batch.cpu().tolist(), **config)
                                                                                 
    def encode(self,
               batch,
               config):
        return self.tokenizer(batch, **config)


paraphrase_model_config = {'num_beams': 1,
                           'encoder_no_repeat_ngram_size': 3,
                           'do_sample': False}

paraphrase_tokenizer_config = {'decode_config': {'skip_special_tokens': True},
                               'encode_config': {'padding': True,
                                                 'return_tensors': 'pt'}}


@pytest.fixture(params=PARAMS)
def create_sampler_scorer_model(request, api_single_param):
    sampler_class, data_path, model_class, model_params = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    scorer = UniversalScorer(metrics=METRICS)
    model_params.update(api_single_param)
    model = model_class(model='GigaChat:latest', **model_params)
    main_metric = 'single_mean_metric_value'
    paraphrase_model = ParaphraseModelHugWrapper('cointegrated/rut5-base-paraphraser')
    paraphrase_tokenizer = ParaphraseTokenizerHugWrapper('cointegrated/rut5-base-paraphraser')
    test_res = valtest_metric_aug_stability(sampler=sampler,
                                            scorer=scorer,
                                            model=model,
                                            main_metric=main_metric,
                                            paraphrase_model=paraphrase_model,
                                            paraphrase_model_config=paraphrase_model_config,
                                            paraphrase_tokenizer=paraphrase_tokenizer,
                                            paraphrase_tokenizer_config=paraphrase_tokenizer_config,
                                            p_augmented=0.1)
    return test_res


def test_valtest_metric_aug_stability(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    main_metric = 'single_mean_metric_value'
    precomputed = test_res['precomputed']
    assert precomputed['augmentations_list'] == ['symbol_swap', 'butter_finger', 'human_level_augmentations', 'word_swap', 'paraphrase', 'asr_augmentations']
    assert any(metric_aug == precomputed['metric_baseline'][main_metric] for metric_aug in list(precomputed['metrics_augmented'][main_metric].values()))


def test_valtest_metric_aug_stability_dict(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    assert isinstance(test_res, dict)


def test_valtest_metric_aug_stability_keys(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_metric_aug_stability_report(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)


def test_valtest_metric_aug_stability_precomputed(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
