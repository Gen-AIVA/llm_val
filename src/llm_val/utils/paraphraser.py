import torch
import typing as tp
from tqdm import trange


def batch_parapraser(batch: tp.List[str],
                     rng: tp.Any,
                     model: tp.Any,
                     tokenizer: tp.Any,
                     model_config: tp.Dict[str, tp.Any],
                     tokenizer_config: tp.Dict[str, tp.Any],
                     p: float=0.02,
                     batch_size: int=64) -> str:
    '''
    Функция для перефразирования предложения
    Parameters:
        batch: Набор предложений (строк)
        rng: Датчик случайных чисел
        model: Языковая модель для перефразирования, имеет методы to_device и generate 
        tokenizer: Соответствующий модели токенизатор, имеет методы encode и decode
        model_config: Параметры языковой модели для перефразирования
        tokenizer_config: Параметры токенизатора
        p: Вероятность того, что предложение будет перефразировано
        batch_size: Размер батча в предложениях
    Return:
        Перефразированное предложение
    '''
    paraphrased, output = [], []
    model.to_device()
    if_paraphrase = rng.random(len(batch)) < p
    batch_2_paraphrase = [text for idx, text in enumerate(batch) if if_paraphrase[idx]]
    for start in trange(0, len(batch_2_paraphrase), batch_size):
        end = start + batch_size if start + batch_size < len(batch_2_paraphrase) else len(batch_2_paraphrase)
        tokenized = tokenizer.encode(batch_2_paraphrase[start:end], tokenizer_config['encode_config'])
        model_config['max_length'] = int(tokenized.input_ids.shape[1] * 1.5 + 10)
        out = model.generate(tokenized.to(model.device), model_config)
        paraphrased.extend([tokenizer.decode(x, tokenizer_config['decode_config']) for x in out])
    parapharsed_idx = 0
    for idx, text in enumerate(batch):
        if if_paraphrase[idx]:
            output.append(paraphrased[parapharsed_idx])
            parapharsed_idx += 1
        else:
            output.append(text)
    torch.cuda.empty_cache()
    return output
