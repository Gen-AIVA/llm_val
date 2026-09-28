import typing as tp

def check_if_correct(answer: str,
                     true_answer: str) -> int:
    '''
    Функция для определения корректности ответа модели для RuBQ-like наборов
    Логика: ответ модели должен входить в правильный ответ
    Parameters:
        answer: Ответ модели
        true_answer: Корректный ответ
    Return:
        1, если ответ корректен, 0 иначе
    '''
    return int(true_answer.lower() in answer.lower())


def check_if_correct_with_options(answer: str,
                                  true_answer: tp.List[str]) -> bool:
    '''
    Функция для определения корректности ответа модели для RuBQ-like наборов
        (при наличии нескольких корректных ответов)
    Логика: ответ модели должен входить в хотя бы один правильный ответ из списка
    Parameters:
        answer: Ответ модели
        true_answer: Список корректных ответов
    Return:
        True, если ответ корректен, False иначе
    '''
    return any(check_if_correct(answer, ta) for ta in true_answer)
