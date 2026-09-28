import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


def call_with_retries(func: tp.Callable[..., tp.Any],
                      timeout: float=60.0,
                      retry_delay: float=0.1,
                      max_attempts: int=3,
                      func_args: tp.Optional[tp.Tuple[tp.Any, ...]]=None,
                      func_kwargs: tp.Optional[tp.Dict[str, tp.Any]]=None) -> tp.Any:
    '''
    Вызывает функцию с повторными попытками и экспоненциальной задержкой
    Parameters:
        func: Вызываемая функция
        timeout: Максимальное время ожидания одного вызова в секундах
        retry_delay: Базовая задержка между повторными попытками в секундах
        max_attempts: Максимальное количество попыток вызова
        func_args: Позиционные аргументы вызываемой функции
        func_kwargs: Именованные аргументы вызываемой функции
    Return:
        Результат вызова функции
    '''
    attempts_count = max(1, max_attempts)
    call_args = func_args or tuple()
    call_kwargs = func_kwargs or {}
    result = None
    last_error: tp.Optional[BaseException] = None
    for attempt_number in range(attempts_count):
        executor: tp.Optional[ThreadPoolExecutor] = None
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(func, *call_args, **call_kwargs)
            result = future.result(timeout=timeout)
            executor.shutdown(wait=False, cancel_futures=True)
            break
        except Exception as error:
            last_error = TimeoutError('Превышено время ожидания вызова') \
                if isinstance(error, FutureTimeoutError) else error
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            if attempt_number < attempts_count - 1:
                time.sleep(retry_delay * 2 ** attempt_number)
    if result is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError('Не удалось получить результат вызова')
    return result
