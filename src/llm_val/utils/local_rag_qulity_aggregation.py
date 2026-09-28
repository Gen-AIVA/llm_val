import typing as tp

def calculate_local_rag_quality(passage_accuracy: tp.List) -> int:
    '''
    Рассчитывает локальное качество RAG для пассажа.
    Parameters:
        passage_accuracy: Список оценок релевантности пассажей.
    Return:
        1, если хотя бы одна оценка равна 1, иначе 0.
    '''
    return 1 if 1 in passage_accuracy else 0


def calculate_local_rag_quality_mean(passage_accuracy: tp.List) -> int:
    '''
    Рассчитывает локальное качество RAG для пассажа.
    Parameters:
        passage_accuracy: Список оценок релевантности пассажей.
    Return:
        Средняя оценка осессоров для пассажей в топе.
    '''
    return max(passage_accuracy)


def calculate_local_rag_quality_max(passage_accuracy: tp.List) -> int:
    '''
    Рассчитывает локальное качество RAG для пассажа.
    Parameters:
        passage_accuracy: Список оценок релевантности пассажей.
    Return:
        Максимальная оценка асессора для пассажа в топе.
    '''
    return max(passage_accuracy)
