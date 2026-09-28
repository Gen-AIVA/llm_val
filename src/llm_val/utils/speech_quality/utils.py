import torch
import typing as tp
from tqdm import trange


def score_data_by_model(text_list: tp.List[str],
                        model: tp.Any,
                        tokenizer: tp.Any,
                        model_config: tp.Dict[str, tp.Any],
                        tokenizer_config: tp.Dict[str, tp.Any],
                        batch_size: int=64) -> tp.List[float]:
    '''
    Вычисляет значения метрики для списка строк с помощью поданной модели 
    Parameters: 
        text_list: Список строк
        model: Языковая модель имеющая методы to_device и generate 
        tokenizer: Соответствующий модели токенизатор, имеет методы encode и decode
        model_config: Параметры языковой модели
        tokenizer_config: Параметры токенизатора
        batch_size: Размер батча
    Return:
        Список значений метрики
    '''
    model.to_device()
    output = []
    for i in trange(0, len(text_list), batch_size):
        start = i
        end = i + batch_size if i + batch_size < len(text_list) else len(text_list)
        batch = text_list[start: end]
        tokenized_input = tokenizer.encode(batch, tokenizer_config['encode_config'])
        model_output = model.generate(tokenized_input, model_config)
        pred = torch.softmax(model_output['logits'], dim=1)
        output += pred[:, 1].detach().cpu().tolist()
    torch.cuda.empty_cache()
    return output
