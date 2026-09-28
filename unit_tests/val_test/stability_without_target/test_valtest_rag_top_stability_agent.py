import pytest
import numpy as np
import pandas as pd
import typing as tp
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, ANY

from llm_val.val_test.stability_without_target.valtest_rag_top_stability_agent import (
    report_valtest_rag_top_stability_agent,
    valtest_rag_top_stability_agent,
    get_augmented_data,
    build_augmented_column,
    apply_augmentation,
    iou_strategy,
    build_rng,
    evaluate_tops
)


class MockChatCompletions:
    def __init__(self, responses_map: tp.Optional[tp.Dict[str, str]] = None):
        self.responses_map = responses_map or {}
        self.call_count = 0
        self.last_model_name = None

    def create(self, model: str, messages: tp.List[tp.Dict[str, str]], **kwargs) -> tp.Any:
        self.call_count += 1
        self.last_model_name = model
        user_content = messages[-1]['content'] if messages else ''
        response_text = user_content
        for key, value in self.responses_map.items():
            if key in user_content:
                response_text = value
                break
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        )


class MockChat:
    def __init__(self, responses_map: tp.Optional[tp.Dict[str, str]] = None):
        self.completions = MockChatCompletions(responses_map)

        
class MockOpenAIClient:
    def __init__(self, responses_map: tp.Optional[tp.Dict[str, str]] = None):
        self.chat = MockChat(responses_map)


class MockSampler:
    def __init__(self, 
                 test_data):
        self.test = {'X': test_data}

        
class MockRAGExecutor:
    def __init__(self, 
                 topK):
        self.params = {'topK': topK}

    def get_parameters(self):
        return self.params

    def get_topK_passages(self, 
                          questions, 
                          topK):
        result = {}
        for question in questions:
            result[question] = {
                'documents': [f'passage{i+1}' for i in range(topK)],
                'embeddings': [np.random.rand(768) for _ in range(topK)]
            }
        return result


@pytest.fixture
def sample_data():
    """Тестовые данные: вопросы и топики."""
    queries = ["Вопрос 1", "Вопрос 2", "Вопрос 3"]
    return pd.DataFrame({"input_query": queries})


@pytest.fixture
def rag_sampler():
    test_data = pd.DataFrame({'input_query': ["Вопрос 1", "Вопрос 2", "Вопрос 3"]})
    return MockSampler(test_data)


@pytest.fixture
def executor_rag():
    return MockRAGExecutor(topK=3)


@pytest.fixture
def iou_symbol_results():
    """Предрасчитанные результаты IOU на уровне символов для topK=1..3."""
    return {1: 0.8, 2: 0.7, 3: 0.6}


@pytest.fixture
def iou_word_results():
    """Предрасчитанные результаты IOU на уровне слов для topK=1..3."""
    return {1: 0.75, 2: 0.65, 3: 0.55}


def test_iou_strategy_full_overlap():
    assert iou_strategy(['p1','p2','p3'], ['p1','p2','p3']) == 1.0
    assert iou_strategy(['p1','p2','p3'], ['p4','p5','p6']) == 0.0
    assert iou_strategy(['p1','p2','p3','p4'], ['p3','p4','p5','p6']) == 2/6


def test_iou_strategy_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        iou_strategy([], ['p1','p2'])


def test_evaluate_tops_empty_input_raises():
    with pytest.raises(ZeroDivisionError):
        evaluate_tops([], [])


def test_report_output_format(iou_symbol_results, iou_word_results):
    """Проверка структуры отчета."""
    report = report_valtest_rag_top_stability_agent(
        iou_symbol_results, iou_word_results, topK=2,
        semaphore_threshold=(0.2, 0.7), is_info=False
    )
    assert isinstance(report, dict)
    assert set(report.keys()) == {"semaphore", "result_plots", "result_dataframes"}
    assert isinstance(report["result_dataframes"], list)
    assert len(report["result_dataframes"]) == 1
    df = report["result_dataframes"][0]
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"Размер топа", "IOU на уровне символов", "IOU на уровне слов"}
    assert df["Размер топа"].tolist() == [1, 2, 3]


def test_report_all_semaphore(iou_symbol_results, iou_word_results):
    """Проверка всех светофоров при различных значениях IOU."""
    iou_sym = {1: 0.8, 2: 0.85}
    iou_wrd = {1: 0.9, 2: 0.95}
    report = report_valtest_rag_top_stability_agent(
        iou_sym, iou_wrd, topK=1, semaphore_threshold=(0.2, 0.7)
    )
    assert report["semaphore"] == "green"

    iou_sym = {1: 0.6, 2: 0.85}
    iou_wrd = {1: 0.3, 2: 0.95}
    report = report_valtest_rag_top_stability_agent(
        iou_sym, iou_wrd, topK=1, semaphore_threshold=(0.2, 0.7)
    )
    assert report["semaphore"] == "yellow"

    iou_sym = {1: 0.1, 2: 0.85}
    iou_wrd = {1: 0.05, 2: 0.95}
    report = report_valtest_rag_top_stability_agent(
        iou_sym, iou_wrd, topK=1, semaphore_threshold=(0.2, 0.7)
    )
    assert report["semaphore"] == "red"


def test_report_semaphore_gray_if_info(iou_symbol_results, iou_word_results):
    """При is_info=True светофор всегда серый."""
    report = report_valtest_rag_top_stability_agent(
        iou_symbol_results, iou_word_results, topK=2, is_info=True
    )
    assert report["semaphore"] == "gray"


@patch("llm_val.val_test.stability_without_target.valtest_rag_top_stability_agent.batch_augmentations")
def test_apply_augmentation_symbol(mock_batch_aug, sample_data):
    texts = sample_data["input_query"].tolist()
    rng = build_rng(42)

    apply_augmentation(texts, "symbol_swap", rng, 0.3)
    mock_batch_aug.assert_called_once_with(
        batch=texts, rng=rng, p=0.3, use_swap=True
    )
    mock_batch_aug.reset_mock()

    apply_augmentation(texts, "butter_finger", rng, 0.5)
    mock_batch_aug.assert_called_once_with(
        batch=texts, rng=rng, p=0.5, use_bf=True
    )
    mock_batch_aug.reset_mock()

    apply_augmentation(texts, "human_level_augmentations", rng, 0.7)
    mock_batch_aug.assert_called_once_with(
        batch=texts, rng=rng, p=0.7,
        use_deletion=True, use_swap=True, use_insert=True, use_bf=True, use_special_symbol=True
    )


def test_apply_augmentation_none(sample_data):
    texts = sample_data["input_query"].tolist()
    rng = build_rng(42)
    result = apply_augmentation(texts, None, rng, 0.3)
    assert result == texts
    assert result is not texts


@patch("llm_val.val_test.stability_without_target.valtest_rag_top_stability_agent.batch_augmentations")
def test_get_augmented_data_symbol(mock_batch_aug, rag_sampler, sample_data):
    def fake_batch_augmentations(batch, rng, p, **kwargs):
        return [f"aug_{i}" for i in range(len(batch))]
    mock_batch_aug.side_effect = fake_batch_augmentations
    aug_list = ["symbol_swap", "butter_finger"]
    aug_col_prefix = 'aug_valtest_rag_top_stability_agent_'
    augmentations_list_with_prefix = list(map(lambda aug: aug_col_prefix + aug, aug_list))

    result = get_augmented_data(
        sampler=rag_sampler,
        data_type="test",
        type_augmentation="symbol",
        augmentations_list=aug_list,
        p_augmented=0.3,
        seed=42,
        exec_env="default"
    )
    assert set(result.columns) == set(augmentations_list_with_prefix)
    assert len(result) == len(sample_data)
    for col in augmentations_list_with_prefix:
        assert result[col].tolist() == [f"aug_{i}" for i in range(len(sample_data))]


@patch("llm_val.val_test.stability_without_target.valtest_rag_top_stability_agent.batch_paraphrase_api")
@patch("llm_val.val_test.stability_without_target.valtest_rag_top_stability_agent.batch_synonym_api")
@patch("llm_val.val_test.stability_without_target.valtest_rag_top_stability_agent.batch_translation_api")
def test_get_augmented_data_word(mock_trans, mock_syn, mock_paraphrase, rag_sampler, sample_data):
    mock_paraphrase.return_value = ["p1", "p2", "p3"]
    mock_syn.return_value = ["s1", "s2", "s3"]
    mock_trans.return_value = ["t1", "t2", "t3"]
    aug_list = ["paraphrase", "synonym", "translation"]
    aug_col_prefix = 'aug_valtest_rag_top_stability_agent_'
    augmentations_list_with_prefix = list(map(lambda aug: aug_col_prefix + aug, aug_list))

    aug_client = MagicMock()
    aug_model_name = "model"
    aug_model_prompts = {"paraphrase": "prompt_p", "synonym": "prompt_s",
                         "translation_fwd": "prompt_f", "translation_bwd": "prompt_b"}

    result = get_augmented_data(
        sampler=rag_sampler,
        data_type="test",
        type_augmentation="word",
        augmentations_list=aug_list,
        p_augmented=0.1,
        seed=42,
        aug_client=aug_client,
        aug_model_name=aug_model_name,
        translation_language="английский",
        word_replace_ratio=0.2,
        max_attempts=5,
        aug_model_prompts=aug_model_prompts,
        exec_env="default"
    )

    assert set(result.columns) == set(augmentations_list_with_prefix)
    assert len(result) == len(sample_data)
    assert result[aug_col_prefix + "paraphrase"].tolist() == ["p1", "p2", "p3"]
    assert result[aug_col_prefix + "synonym"].tolist() == ["s1", "s2", "s3"]
    assert result[aug_col_prefix + "translation"].tolist() == ["t1", "t2", "t3"]

    mock_paraphrase.assert_called_once_with(
        batch=sample_data["input_query"].tolist(),
        aug_client=aug_client,
        aug_model_name=aug_model_name,
        system_prompt="prompt_p",
        max_attempts=5
    )
    mock_syn.assert_called_once_with(
        batch=sample_data["input_query"].tolist(),
        aug_client=aug_client,
        aug_model_name=aug_model_name,
        rng=ANY,
        word_replace_ratio=0.2,
        system_prompt="prompt_s",
        max_attempts=5
    )
    mock_trans.assert_called_once_with(
        batch=sample_data["input_query"].tolist(),
        aug_client=aug_client,
        aug_model_name=aug_model_name,
        target_language="английский",
        system_prompt_fwd="prompt_f",
        system_prompt_bwd="prompt_b",
        max_attempts=5
    )


def test_get_augmented_data_unknown_augmentation(rag_sampler):
    with pytest.raises(ValueError, match="Неизвестный тип аугментации"):
        get_augmented_data(
            sampler=rag_sampler,
            data_type="test",
            type_augmentation="word",
            augmentations_list=["unknown_aug"],
            aug_client=MagicMock(),
            aug_model_name="model",
            exec_env="default"
        )


@patch("llm_val.val_test.stability_without_target.valtest_rag_top_stability_agent.get_augmented_data")
@patch("llm_val.val_test.stability_without_target.valtest_rag_top_stability_agent.evaluate_tops")
def test_valtest_with_precomputed(mock_evaluate, mock_get_augmented, rag_sampler, executor_rag,
                                  iou_symbol_results, iou_word_results):
    result = valtest_rag_top_stability_agent(
        sampler=rag_sampler,
        executor=executor_rag,
        semaphore_threshold=(0.2, 0.7),
        iou_symbol_results=iou_symbol_results,
        iou_word_results=iou_word_results,
        topK=3
    )
    mock_get_augmented.assert_not_called()
    mock_evaluate.assert_not_called()

    precomputed = result["precomputed"]
    assert precomputed["iou_symbol_results"] == iou_symbol_results
    assert precomputed["iou_word_results"] == iou_word_results

    report = result["report"]
    assert report["semaphore"] == "yellow"