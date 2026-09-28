import pytest
import joblib
import typing as tp
from nlpaug.augmenter.char.keyboard import KeyboardAug
from nlpaug.augmenter.char.random import RandomCharAug
from transformers import T5ForConditionalGeneration, T5Tokenizer

from llm_val.model.BaseGigaModel import BaseGigaModel
from llm_val.sampler.DialogueSampler import DialogueSampler
from llm_val.val_test.specification_chat.valtest_adversarial_stability import valtest_adversarial_stability

DATA_PATH = './data/joy_dataset.pkl'
STOP_PHRASES_PATH = './data/stop_phrases_ex.pkl'
PARAPHRASER_PARAMS = {'num_beams': 10,
                      'encoder_no_repeat_ngram_size': 4,
                      'do_sample': True}
TOKENIZER_PARAMS = {'return_tensors': 'pt',
                    'padding': True}
PARAPHRASER_PATH = 'cointegrated/rut5-base-paraphraser'
AUGMENTS = [KeyboardAug(lang='ru', 
                        aug_char_p=0.2, 
                        aug_word_p=0.2, 
                        aug_char_max=2, 
                        aug_word_max=2),
            RandomCharAug(action='delete', 
                          aug_char_p=0.2, 
                          aug_word_p=0.2, 
                          aug_char_max=2, 
                          aug_word_max=2, 
                          name='RandomCharDelete_Aug'),
            RandomCharAug(action='swap', 
                          aug_char_p=0.2, 
                          aug_word_p=0.2, 
                          aug_char_max=2, 
                          aug_word_max=2, 
                          name='RandomCharSwap_Aug')]

class Paraphraser:
    '''
    Обёртка для модели перефразирования
    '''
    def __init__(self, 
                 paraphraser_path: str,
                 paraphraser_param: tp.Dict[str, tp.Any], 
                 tokenizer_param: tp.Dict[str, tp.Any],
                 device: str='cuda'):
        '''
        paraphraser_path: Путь до модели
        paraphraser_param: Словарь с параметрами модели
        tokenizer_param: Словарь с параметрами токенизатора
        device: CPU или CUDA
        '''
        self.tokenizer = T5Tokenizer.from_pretrained(paraphraser_path)
        self.model = T5ForConditionalGeneration.from_pretrained(paraphraser_path)
        self.params = {'model': paraphraser_param,
                       'tokenizer': tokenizer_param}
        self.model.to(device)
        self.model.eval()

    
    def paraphrase(self, 
                   sentence: str) -> str:
        '''
        Функция для перефразирования
        Parameters:
            sentence: Предложение
        Return:
            Перефразированное предложение
        '''
        model_input = self.tokenizer(sentence, **self.params['tokenizer']).to(self.model.device)
        max_size = int(model_input.input_ids.shape[1] * 1.5 + 10)
        self.params['model']['max_length'] = max_size
        output = self.model.generate(**model_input, **self.params['model'])
        output_decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_decoded
    

@pytest.fixture()
def create_test(request, api_single_param):
    data = joblib.load(DATA_PATH)
    sampler = DialogueSampler(data)
    model = BaseGigaModel(**api_single_param)
    model_opp = BaseGigaModel(**api_single_param)
    paraphraser = Paraphraser(PARAPHRASER_PATH,
                              PARAPHRASER_PARAMS,
                              TOKENIZER_PARAMS)
    res = valtest_adversarial_stability(sampler=sampler,
                                        model=model,
                                        model_opponent=model_opp,
                                        stop_phrases_path=STOP_PHRASES_PATH,
                                        paraphraser=paraphraser,
                                        augments=AUGMENTS,
                                        data_type='test',
                                        epsilon=0.3,
                                        gamma=0.99,
                                        bound=(1, 3),
                                        max_iter=3,
                                        threshold=0.08,
                                        seed=42,
                                        time_step=7,
                                        precomputed=None)
    return res
    


def test_valtest_adversarial_stability_result(create_test):
    res = create_test
    assert isinstance(res, dict)
    assert set(res.keys()) == {'report', 'precomputed'}


def test_valtest_adversarial_stability_report(create_test):
    res = create_test
    assert isinstance(res['report'], dict)
    assert set(res['report'].keys()) == {'semaphore', 'result_dataframes', 'result_plots'}


def test_valtest_adversarial_stability_precomputed(create_test):
    res = create_test
    assert isinstance(res['precomputed'], dict)
    assert set(res['precomputed'].keys()) == {'1', '2', 'draw'}