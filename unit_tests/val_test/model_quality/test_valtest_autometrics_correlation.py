import pytest
import joblib
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.val_test.model_quality.valtest_autometrics_correlation import valtest_autometrics_correlation

PARAMS = [(['bleu', 'meteor', 'rouge', 'cosine_dist', 'llm_as_judge'], 0.2)]

class GigaChatJudge:
    def __init__(self, model_name: str = "GigaChat-2-Max"):
        self.GIGA_CONFIG = {
            "auth_url": 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
            "credentials": 'AUTH_DATA',
            "model": model_name,
            "scope": "GIGACHAT_API_CORP",
            "verify_ssl_certs": False,
            "profanity_check": False,
        }
        self.client = GigaChat(**self.GIGA_CONFIG)
        self.client.__enter__()

    def predict(self, references, prediction, **kwargs) -> str:
        """
        Отправляет запрос в GigaChat для оценки ответа.
        Формирует промпт внутри.
        """
        prompt = f"""
            Ты являешься экспертом по оценке текстовых ответов.
            Твоя задача — сравнить предсказанный ответ с набором эталонных (reference) ответов.

            Эталонные ответы:
            {chr(10).join(references)}

            Предсказанный ответ:
            {prediction}

            Оцени, насколько предсказание соответствует смыслу эталонных ответов.
            Используй шкалу от 1 до 5, где:
            - 5 = идеальное совпадение,
            - 4 = почти идеально,
            - 3 = частично верно,
            - 2 = почти неверно,
            - 1 = совершенно неверно.

            Возвращай только одно число от 1 до 5, без комментариев.
        """
        for _ in range(10):
            response = self.client.chat(
                Chat(
                    messages=[Messages(role=MessagesRole.USER, content=prompt)],
                    temperature=0.0,
                )
            )
            try:
                score = float(response.choices[0].message.content.strip().split()[0])
                return score
            except ValueError:
                print('Не удалось извлечь числовой score из ответа модели, пробуем снова...')
                response = self.client.chat(
                    Chat(
                        messages=[Messages(role=MessagesRole.USER, content=prompt)],
                        temperature=0.0,
                    )
                )
        return 0.0

@pytest.fixture(params=PARAMS)
def create_scorer_sampler(request):
    metrics_to_calc, corr_threshold = request.param
    data_rubq = joblib.load('data/data_rubq.pkl')
    data_la = joblib.load('data/data_longanswer.pkl')
    sampler_rubq = RuBQSampler(data_rubq)
    sampler_la = LongAnswerSampler(data_la)
    rubq_test = getattr(sampler_rubq, 'test')
    rubq_test['y'] = getattr(sampler_la, 'test')['y']
    setattr(sampler_rubq, 'test', rubq_test)

    giga_judge = GigaChatJudge()

    res = valtest_autometrics_correlation(sampler=sampler_rubq,
                                          metrics_to_calc=metrics_to_calc,
                                          corr_threshold=corr_threshold,
                                          data_type='test',
                                          judge_model=giga_judge)
    return res, request.param, sampler_rubq


def test_valtest_autometrics_correlation_report(create_scorer_sampler):
    test_res, params, sampler = create_scorer_sampler
    metrics_to_calc, _ = params
    report = test_res['report']
    y = getattr(sampler, 'test')['y']
    
    for df in report['result_dataframes']:
        assert set(df.index) == set(metrics_to_calc)    
    assert len(report['result_dataframes']) == len(y.columns)


def test_valtest_autometrics_correlation_precomputed(create_scorer_sampler):
    test_res, params, _ = create_scorer_sampler
    metrics_to_calc, _ = params
    precomputed = test_res['precomputed']
    
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'corr_values'}
    assert set(precomputed['corr_values'].index) == set(metrics_to_calc)
