import re
import time
import socks
import random
import socket
import requests
import numpy as np
import pandas as pd
import typing as tp
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from fake_useragent import UserAgent
from requests.models import Response

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.ssl_context import no_ssl_verification
from llm_val.utils.report_helper import (semaphore_by_threshold_without_yellow,
                                         semaphore_by_threshold_without_red,
                                         worst_semaphore)


def report_valtest_internet_exact_match(total_dict_without_answers: tp.Dict[str, tp.List],
                                        total_dict_with_answers: tp.Optional[tp.Dict[str, tp.List]],
                                        semaphore_threshold_with_answers: float=0.05,
                                        semaphore_threshold_without_answers: float=0.05,
                                        data_type: str='test',
                                        is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        total_dict_without_answers: Словарь с результатом поиска вопросов без ответов
        total_dict_with_answers: Словарь с результатом поиска вопросов с ответами
        semaphore_threshold_with_answers: Порог красного светофора (при использовании ответов)
        semaphore_threshold_without_answers: Порог желтого светофора (без использования ответов)
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    color_list = []
    result_dataframes = []
    result_dataframes.append(pd.DataFrame({'Доля найденных вопросов без ответов': [total_dict_without_answers['found_frac']]}))
    result_dataframes.append(pd.DataFrame({'Найденные вопросы': total_dict_without_answers['found_questions'],
                                           'Найденные ссылки': total_dict_without_answers['found_urls']}))
    color_list.append(semaphore_by_threshold_without_red(total_dict_without_answers['found_frac'],
                                                         semaphore_threshold_without_answers,
                                                         greater_is_better=False,
                                                         border_is_bigger=False))
    if total_dict_with_answers is not None:
        result_dataframes.append(pd.DataFrame({'Доля найденных вопросов с ответами': [total_dict_with_answers['found_frac']]}))
        result_dataframes.append(pd.DataFrame({'Найденные вопросы': total_dict_with_answers['found_questions'],
                                               'Найденные ответы': total_dict_with_answers['found_answers'],
                                               'Найденные ссылки': total_dict_with_answers['found_urls']}))
        color_list.append(semaphore_by_threshold_without_yellow(total_dict_with_answers['found_frac'],
                                                                semaphore_threshold_with_answers,
                                                                greater_is_better=False,
                                                                border_is_bigger=False))
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else worst_semaphore(color_list)
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = result_dataframes
    return res_dict
    
    
def valtest_internet_exact_match(sampler: Sampler,
                                 data_type: str='test',
                                 semaphore_threshold_with_answers: float=0.05,
                                 semaphore_threshold_without_answers: float=0.05,
                                 top_to_check: int=1,
                                 engine: str='rambler',
                                 queries_before_timeout: int=100,
                                 long_timeout: float=60.,
                                 short_timeout: float=3.,
                                 min_short_timeout: float=1.1,
                                 response_waiting_time: float=10.,
                                 proxying_policy: str='none',
                                 max_proxy_retries: int=5,
                                 http_proxies: tp.Optional[tp.List]=None,
                                 socks5_proxies: tp.Optional[tp.List]=None,
                                 socks4_proxies: tp.Optional[tp.List]=None,
                                 random_state: tp.Optional[int]=None,
                                 verbose: bool=False,
                                 is_info: bool=False,
                                 found_frac_without_answers: tp.Optional[float]=None,
                                 found_frac_with_answers: tp.Optional[float]=None,
                                 **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на поиск точного совпадения текста вопроса (при наличии - и ответа) в интернете.
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold_with_answers: Порог красного светофора (при использовании ответов)
        semaphore_threshold_without_answers: Порог желтого светофора (без использования ответов)
        top_to_check: Число найденных ссылок, которые нужно выдать по итогам поиска (если столько имеется)
        engine: Поисковой движок; доступны 'google', 'duckduckgo', 'yandex' и 'rambler' (gigasearch proxy)
        queries_before_timeout: Число запросов до длинного таймаута для избежания блокировки
        long_timeout: Среднее время длинного таймаута между каждыми queries_before_timeout запросами в секундах
        short_timeout: Среднее время таймаута между двумя запросами подряд в секундах
        min_short_timeout: Минимальное время таймаута между двумя запросами подряд в секундах.
                           Для rambler не рекомендуется ставить значение меньше 1, т.к. это приводит к бану IP.
        response_waiting_time: Время ожидания ответа от поисковика
        proxying_policy: Тип проксирования, может быть none, random, http, socks5, socks4
                       none - не использовать проксирование, есть риск быть забаненным в гугле
                       random - случайный тип проксирования; 
                                вероятность выбрать тип прокси равна доле прокси этого типа среди всех прокси
                       http - только http прокси
                       socks5 - только socks5 прокси
                       socks4 - только socks4 прокси
        max_proxy_retries: Количество попыток получить ответ с различными прокси
        http_proxies: Список ссылок на http-прокси
        socks5_proxies: Список ссылок на socks5-прокси
        socks4_proxies: Список ссылок на socks4-прокси
        random_state: Рандом стейт для выбора прокси, необходим для дебага неработающих прокси.
                      НЕ ЗАКРЕПЛЯТЬ в обычном использовании, т.к. это приводит к использованию одних и тех же прокси, 
                      что может снизить их срок жизни.
        is_info: True, если тест информативный
        found_frac_without_answers: Опциональное значение доли найденных в интернете вопросов без ответов.
            Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
        found_frac_with_answers: Опциональное значение доли найденных в интернете вопросов с ответами.
            Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    data = getattr(sampler, data_type)['X']
    failed_questions = []
    failed_pairs = []
    question_list = data['question'].to_list()
    if found_frac_without_answers is None:
        total_dict_without_answers, failed_questions = find_questions_without_answers(question_list,
                                                                                      top_to_check=top_to_check,
                                                                                      engine=engine,
                                                                                      queries_before_timeout=queries_before_timeout,
                                                                                      long_timeout=long_timeout,
                                                                                      short_timeout=short_timeout,
                                                                                      min_short_timeout=min_short_timeout,
                                                                                      response_waiting_time=response_waiting_time,
                                                                                      proxying_policy=proxying_policy,
                                                                                      max_proxy_retries=max_proxy_retries,
                                                                                      http_proxies=http_proxies,
                                                                                      socks5_proxies=socks5_proxies,
                                                                                      socks4_proxies=socks4_proxies,
                                                                                      random_state=random_state,
                                                                                      verbose=verbose)
        if verbose and len(failed_questions) > 0:
            print(f'{len(failed_questions)} questions did not get a response from serach')
        found_frac_without_answers = total_dict_without_answers['found_frac']
    else:
        total_dict_without_answers = {'found_questions': [],
                                      'found_urls': [],
                                      'found_frac': found_frac_without_answers}
    if found_frac_with_answers is None:
        total_dict_with_answers = None
        if not data['true_answer'].isnull().all():
            if data['true_answer'].map(type).eq(str).all():
                answer_list = data.apply(lambda row: [row['answer_options'][row['true_answer']]], axis=1).to_list()
            else:
                answer_list = data['true_answer'].to_list()
            total_dict_with_answers, failed_pairs = find_questions_with_answers(question_list,
                                                                                answer_list,
                                                                                top_to_check=top_to_check,
                                                                                engine=engine,
                                                                                queries_before_timeout=queries_before_timeout,
                                                                                long_timeout=long_timeout,
                                                                                short_timeout=short_timeout,
                                                                                min_short_timeout=min_short_timeout,
                                                                                response_waiting_time=response_waiting_time,
                                                                                proxying_policy=proxying_policy,
                                                                                max_proxy_retries=max_proxy_retries,
                                                                                http_proxies=http_proxies,
                                                                                socks5_proxies=socks5_proxies,
                                                                                socks4_proxies=socks4_proxies,
                                                                                random_state=random_state,
                                                                                verbose=verbose)
            if verbose and len(failed_pairs) > 0:
                print(f'{len(failed_pairs)} question-answer pairs did not get a response from serach')
            found_frac_with_answers = total_dict_with_answers['found_frac']
    else: 
        total_dict_with_answers = {'found_questions': [],
                                   'found_answers': [],
                                   'found_urls': [],
                                   'found_frac': found_frac_with_answers}
        
    report = report_valtest_internet_exact_match(total_dict_without_answers,
                                                 total_dict_with_answers,
                                                 semaphore_threshold_with_answers,
                                                 semaphore_threshold_without_answers,
                                                 data_type,
                                                 is_info)
    precomputed = {'found_frac_without_answers': found_frac_without_answers,
                   'found_frac_with_answers': found_frac_with_answers,
                   'failed_questions': failed_questions,
                   'failed_pairs': failed_pairs}
    return {'report': report,
            'precomputed': precomputed}


def find_questions_with_answers(question_list: tp.List[str],
                                answer_list: tp.List[tp.List[str]],
                                top_to_check: int=1,
                                engine: str='google',
                                queries_before_timeout: int=100,
                                long_timeout: float=60.,
                                short_timeout: float=3.,
                                min_short_timeout: float=1.,
                                response_waiting_time: float=10.,
                                proxying_policy: str='none',
                                max_proxy_retries: int=5,
                                http_proxies: tp.Optional[tp.List]=None,
                                socks5_proxies: tp.Optional[tp.List]=None,
                                socks4_proxies: tp.Optional[tp.List]=None,
                                random_state: tp.Optional[int]=None,
                                verbose: bool=False) -> tp.Tuple[tp.Dict[str, tp.Any], tp.List[tp.Tuple[str, str]]]:
    '''
    Функция возвращает результат поиска вопросов вместе с ответами
    Parameters:
        question_list: Список с вопросами
        answer_list: Список с ответами
        top_to_check: Число ссылок, которые нужно выдать по итогам поиска (при наличии)
        engine: Поисковой движок; доступны 'google', 'duckduckgo', 'yandex' и 'rambler' (gigasearch)
        queries_before_timeout: Число запросов до длинного таймаута для избежания блокировки
        long_timeout: Среднее время длинного таймаута между каждыми queries_before_timeout запросами в секундах
        short_timeout: Среднее время таймаута между двумя запросами подряд в секундах
        min_short_timeout: Минимальное время таймаута между двумя запросами подряд в секундах.
                           Для rambler не рекомендуется ставить значение меньше 1, т.к. это приводит к бану IP.
        response_waiting_time: Время ожидания ответа от поисковика
        proxying_policy: Тип проксирования, может быть none, random, http, socks5, socks4
                       none - не использовать проксирование, есть риск быть забаненным в гугле
                       random - случайный тип проксирования; 
                                вероятность выбрать тип прокси равна доле прокси этого типа среди всех прокси
                       http - только http прокси
                       socks5 - только socks5 прокси
                       socks4 - только socks4 прокси
        max_proxy_retries: Количество попыток получить ответ с различными прокси
        http_proxies: Список ссылок на http-прокси
        socks5_proxies: Список ссылок на socks5-прокси
        socks4_proxies: Список ссылок на socks4-прокси
        random_state: Рандом стейт для выбора прокси, необходим для дебага неработающих прокси.
                      НЕ ЗАКРЕПЛЯТЬ в обычном использовании, т.к. это приводит к использованию одних и тех же прокси, 
                      что может снизить их срок жизни.
    Return:
        Словарь с найденными вопросами, ответами, ссылками и долей найденных вопросов с ответами
        Список Tuple (question, answer) с запросами, на которые ответа получить не удалось
    '''
    found_questions, found_answers, found_urls = [], [], []
    queries_remaining_before_timeout = queries_before_timeout
    failed_pairs = []
    iterator = tqdm(range(len(question_list))) if verbose else range(len(question_list))
    max_short_timeout = 2 * short_timeout - min_short_timeout
    for cur_iter in iterator:
        for cur_answer_num in range(len(answer_list[cur_iter])):
            cur_urls, failed = get_urls_from_search(question_list[cur_iter],
                                                    answer_list[cur_iter][cur_answer_num],
                                                    top_to_check=top_to_check,
                                                    engine=engine,
                                                    response_waiting_time=response_waiting_time,
                                                    proxying_policy=proxying_policy,
                                                    max_proxy_retries=max_proxy_retries,
                                                    http_proxies=http_proxies,
                                                    socks5_proxies=socks5_proxies,
                                                    socks4_proxies=socks4_proxies,
                                                    random_state=random_state,
                                                    verbose=verbose)
            queries_remaining_before_timeout -= 1
            if failed:
                failed_pairs.append((question_list[cur_iter], answer_list[cur_iter][cur_answer_num]))
            elif len(cur_urls) > 0:
                found_questions.append(question_list[cur_iter])
                found_answers.append(answer_list[cur_iter][cur_answer_num])
                found_urls.append(cur_urls)
                break

            time.sleep(random.uniform(min_short_timeout, max_short_timeout))
            if queries_remaining_before_timeout == 0:
                time.sleep(long_timeout+random.random()*long_timeout/10-long_timeout/20)
                queries_remaining_before_timeout = queries_before_timeout
    return {'found_questions': found_questions,
            'found_answers': found_answers,
            'found_urls': found_urls,
            'found_frac': len(found_questions) / len(question_list)}, \
           failed_pairs


def find_questions_without_answers(question_list: tp.List[str],
                                   top_to_check: int=1,
                                   engine: str='google',
                                   queries_before_timeout: int=100,
                                   long_timeout: float=60.,
                                   short_timeout: float=5.,
                                   min_short_timeout: float=1.,
                                   response_waiting_time: float=10.,
                                   proxying_policy: str='none',
                                   max_proxy_retries: int=5,
                                   http_proxies: tp.Optional[tp.List]=None,
                                   socks5_proxies: tp.Optional[tp.List]=None,
                                   socks4_proxies: tp.Optional[tp.List]=None,
                                   random_state: tp.Optional[int]=None,
                                   verbose: bool=False) -> tp.Tuple[tp.Dict[str, tp.Any], tp.List[tp.Tuple[str, str]]]:
    '''
    Функция возвращает результат поиска вопросов без ответов
    Parameters:
        question_list: Список с вопросами
        top_to_check: Число ссылок, которые нужно выдать по итогам поиска (при наличии)
        engine: Поисковой движок; доступны 'google', 'duckduckgo', 'yandex' и 'rambler' (gigasearch)
        queries_before_timeout: Число запросов до длинного таймаута для избежания блокировки
        long_timeout: Среднее время длинного таймаута между каждыми queries_before_timeout запросами в секундах
        short_timeout: Среднее время таймаута между двумя запросами подряд в секундах
        min_short_timeout: Минимальное время таймаута между двумя запросами подряд в секундах.
                           Для rambler не рекомендуется ставить значение меньше 1, т.к. это приводит к бану IP.
        response_waiting_time: Время ожидания ответа от поисковика в секундах
        proxying_policy: Тип проксирования, может быть none, random, http, socks5, socks4
                       none - не использовать проксирование, есть риск быть забаненным в гугле
                       random - случайный тип проксирования; 
                                вероятность выбрать тип прокси равна доле прокси этого типа среди всех прокси
                       http - только http прокси
                       socks5 - только socks5 прокси
                       socks4 - только socks4 прокси
        max_proxy_retries: Количество попыток получить ответ с различными прокси
        http_proxies: Список ссылок на http-прокси
        socks5_proxies: Список ссылок на socks5-прокси
        socks4_proxies: Список ссылок на socks4-прокси
        random_state: Рандом стейт для выбора прокси, необходим для дебага неработающих прокси.
                      НЕ ЗАКРЕПЛЯТЬ в обычном использовании, т.к. это приводит к использованию одних и тех же прокси, 
                      что может снизить их срок жизни.
    Return:
        Словарь с найденными вопросами, ссылками и долей найденных вопросов
        Список Tuple (question,) вопросов, на которые ответа получить не удалось
    '''
    found_questions, found_urls = [], []
    queries_remaining_before_timeout = queries_before_timeout
    failed_questions = []
    iterator = tqdm(question_list) if verbose else question_list
    max_short_timeout = 2 * short_timeout - min_short_timeout
    for cur_question in iterator:
        cur_urls, failed = get_urls_from_search(cur_question,
                                                top_to_check=top_to_check,
                                                engine=engine,
                                                response_waiting_time=response_waiting_time,
                                                proxying_policy=proxying_policy,
                                                max_proxy_retries=max_proxy_retries,
                                                http_proxies=http_proxies,
                                                socks5_proxies=socks5_proxies,
                                                socks4_proxies=socks4_proxies,
                                                random_state=random_state,
                                                verbose=verbose)
        queries_remaining_before_timeout -= 1
        if failed:
            failed_questions.append((cur_question,))
        elif len(cur_urls) > 0:
            found_questions.append(cur_question)
            found_urls.append(cur_urls)

        time.sleep(random.uniform(min_short_timeout, max_short_timeout))
        if queries_remaining_before_timeout == 0:
            time.sleep(long_timeout+random.random()*long_timeout/10-long_timeout/20)
            queries_remaining_before_timeout = queries_before_timeout
    return {'found_questions': found_questions,
            'found_urls': found_urls,
            'found_frac': len(found_questions) / len(question_list)}, \
           failed_questions
    
    
def get_urls_from_search(question: str,
                         answer: tp.Optional[str]=None,
                         top_to_check: int=1,
                         engine: str='google',
                         response_waiting_time: float=10.,
                         proxying_policy: str='none',
                         max_proxy_retries: int=5,
                         http_proxies: tp.Optional[tp.List]=None,
                         socks5_proxies: tp.Optional[tp.List]=None,
                         socks4_proxies: tp.Optional[tp.List]=None,
                         random_state: tp.Optional[int]=None,
                         verbose: bool=False) -> tp.Tuple[tp.List[str], bool]:
    '''
    Функция делает запрос к поисковой системе через requests.get (что должно помогать избегать блокировки гуглом)
        и выдает все top_to_check ссылок, в которых есть точное совпадение с текстом query.
    Parameters:
        question: Вопрос для поиска в интернете
        answer: Ответ для поиска в интернете
        top_to_check: Число ссылок, которые нужно выдать по итогам поиска (при наличии)
        engine: Поисковой движок; доступны 'google', 'duckduckgo', 'yandex' и 'rambler' (gigasearch)
        response_waiting_time: Время ожидания ответа от поисковика в секундах
        proxying_policy: Тип проксирования, может быть none, random, http, socks5, socks4
                       none - не использовать проксирование, есть риск быть забаненным в гугле
                       random - случайный тип проксирования; 
                                вероятность выбрать тип прокси равна доле прокси этого типа среди всех прокси
                       http - только http прокси
                       socks5 - только socks5 прокси
                       socks4 - только socks4 прокси
        max_proxy_retries: Количество попыток получить ответ с различными прокси
        http_proxies: Список ссылок на http-прокси
        socks5_proxies: Список ссылок на socks5-прокси
        socks4_proxies: Список ссылок на socks4-прокси
        random_state: Рандом стейт для выбора прокси, необходим для дебага неработающих прокси.
                      НЕ ЗАКРЕПЛЯТЬ в обычном использовании, т.к. это приводит к использованию одних и тех же прокси, 
                      что может снизить их срок жизни.
    Return:
        Список сайтов с найденными вопросом и ответом (пустой список при отстутсвии/неудачном запросе)
        Флаг, означающий, что ответ на запрос не был получен
    '''
    rambler_proxy_url = 'https://gigachat.dev.app.sberdevices.ru/rambler_proxy'
    yandex_domain = ['ya.ru', 'yandex.com.ru', 'yandex.by']
    google_domain = ['google.ru', 'google.com', 'google.es', 'google.it', 'google.fr']
    
    headers = {'User-Agent': get_random_user_agent()}
    proxies_to_use = {'http': http_proxies if http_proxies else [],
                      'socks5': socks5_proxies if socks5_proxies else [],
                      'socks4': socks4_proxies if socks4_proxies else []}
    query = f"%22{question.replace(' ', '%20')}%22"
    if answer:
        query += f"%20%22{answer.replace(' ', '%20')}%22"
    urls = []
    failed = False
    with no_ssl_verification():
        if engine == 'google':
            cur_domain = random.choice(google_domain)
            URL = f'https://www.{cur_domain}/search?q={query}&tbs=li:1&hl=ru'
            page = get_page(URL=URL,
                            headers=headers,
                            proxies_to_use=proxies_to_use,
                            proxying_policy=proxying_policy,
                            max_proxy_retries=max_proxy_retries,
                            timeout=response_waiting_time,
                            verbose=verbose)
            if page:
                urls = parse_google_search_page(page)
                page.cookies.clear()
            else:
                failed = True
                if verbose: 
                    print('No pages scrapped')
                    
        elif engine == 'rambler':
            URL = f'{rambler_proxy_url}?text={query}'
            page = get_page(URL=URL,
                            headers=headers,
                            proxies_to_use=proxies_to_use,
                            proxying_policy=proxying_policy,
                            max_proxy_retries=max_proxy_retries,
                            timeout=response_waiting_time,
                            verbose=verbose,
                            is_rambler=True)
            if page:
                urls = parse_rambler_search_page(page)
                page.cookies.clear()
            else:
                failed = True
                if verbose: 
                    print('No pages scrapped')  
                    
        elif engine == 'yandex':
            cur_domain = random.choice(yandex_domain)
            URL = f'https://www.{cur_domain}/search?text={query}&search_source=dzen_desktop_safe'
            page = get_page(URL=URL,
                            headers=headers,
                            proxies_to_use=proxies_to_use,
                            proxying_policy=proxying_policy,
                            max_proxy_retries=max_proxy_retries,
                            timeout=response_waiting_time,
                            verbose=verbose)   
            if page:      
                urls = parse_yandex_search_page(page)
                page.cookies.clear()
            else:
                failed = True
                if verbose: 
                    print('No pages scrapped')

        elif engine == 'duckduckgo':
            URL = f'https://html.duckduckgo.com/html/?q={query}'
            page = get_page(URL=URL,
                            headers=headers,
                            proxies_to_use=proxies_to_use,
                            proxying_policy=proxying_policy,
                            max_proxy_retries=max_proxy_retries,
                            timeout=response_waiting_time,
                            verbose=verbose)   
            if page:
                urls = parse_ddg_search_page(page)
                page.cookies.clear()
            else:
                failed = True
                if verbose: 
                    print('No pages scrapped')        
    return urls[:top_to_check], failed


def get_page(URL: str,
             headers: tp.Dict[str, tp.Any],
             proxies_to_use: tp.Optional[tp.Dict[str, str]]=None,
             proxying_policy: str='none',
             max_proxy_retries: int=5,
             timeout: int=10,
             verbose: bool=False,
             random_state: tp.Optional[int]=None,
             is_rambler=False) -> tp.Any:
    '''
    Получение HTML-страницы по ссылке
    Parameters:
        URL: Ссылка на страницу
        headers: Header запроса
        proxies_to_use: Словарь со списками прокси каждого типа
        proxying_policy: Тип проксирования, может быть none, random, http, socks5, socks4
        max_proxy_retries: Максимальное число прокси для попытки послать запрос
        timeout: Время ожидания ответа от страницы
        verbose: Если True, то показываем прогресс/дебаг
        random_state: Рандом стейт для выбора прокси, необходим для дебага неработающих прокси
        is_rambler: True, если используется rambler
    Return:
        HTML страница; None, если ответ не пришел или код ответа != 200
    '''    
    if random_state:
        random.seed(random_state)
    
    page = None
    if is_rambler:
        tokens = get_rambler_token()
        for token in tokens:
            current_headers = headers.copy()
            current_headers['Authorization'] = token
            try:
                page = requests.request("GET", URL, headers=current_headers, data={})
                if page.status_code != 200:
                    page = None
                    if verbose:
                        print(f"Rambler request failed with token, status code: {page.status_code}")
                else:
                    break
            except Exception as e:
                if verbose:
                    print(f'Rambler request failed with token: {str(token)}, error: {str(e)}')

    elif proxying_policy != 'none':
        for retry in range(max_proxy_retries):
            if proxying_policy == 'random':
                total_proxies = sum([len(v) for v in proxies_to_use.values()])
                proxy_type = random.choices(list(proxies_to_use.keys()), k=1, weights=(len(v) / total_proxies for _, v in proxies_to_use.items()))[0]
            else:
                proxy_type = proxying_policy 
            proxy_list = proxies_to_use[proxy_type]
            proxy = random.choice(proxy_list)
            page = get_response_with_proxy(URL=URL,
                                           headers=headers,
                                           proxy=proxy,
                                           proxy_type=proxy_type,
                                           timeout=timeout,
                                           verbose=verbose)
            if page:
                if page.status_code == 200:
                    break
                elif verbose:
                    print(f'DEBUG: Got status code {page.status_code} for proxy {p}')         
    else:
        try:
            page = requests.get(URL, timeout=timeout, headers=headers)
        except:
            page = None         
    if page and page.status_code != 200:
        if verbose:
            print(f'DEBUG: Got final status code {page.status_code}, returning empty page')  
        page = None       
    return page

def get_response_with_proxy(URL: str,
                            headers: tp.Dict[str, tp.Any],
                            proxy: str,
                            proxy_type: str,
                            timeout: int=10,
                            verbose: bool=False) -> tp.Any:
    '''
    Функция для получения response'а через прокси по ссылке 
    Parameters: 
        URL: Ссылка на страницу
        headers: Header запроса
        proxy: Прокси для запроса
        proxy_type: Тип прокси, может быть 'http', 'socks5', 'socks4'
        timeout: Время ожидания ответа от страницы
        verbose: Если True, то показываем прогресс/дебаг
    Return:
        Ответ от requests.get; None, если ответа не пришло
    '''
    page = None
    if proxy_type == 'http':
        proxies = {'http': proxy,
                   'https': proxy}
        try:
            page = requests.get(URL, timeout=timeout, headers=headers, proxies=proxies)
        except:
            if verbose:
                print(f'Proxy {proxy} not available')
    
    elif proxy_type == 'socks5':
        ip = proxy.split(':')[0]
        port = proxy.split(':')[1] if len(proxy.split(':')) > 1 else 0
        socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, ip, int(port))
        socket.socket = socks.socksocket
        try:
            page = requests.get(URL, timeout=timeout, headers=headers)
        except:
            if verbose:
                print(f'Proxy {proxy} not available')

    elif proxy_type == 'socks4':
        ip = proxy.split(':')[0]
        port = proxy.split(':')[1] if len(proxy.split(':')[1]) > 1 else 0
        socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS4, ip, int(port))
        socket.socket = socks.socksocket
        try:
            page = requests.get(URL, timeout=timeout, headers=headers)
        except:
            if verbose:
                print(f'Proxy {proxy} not available')      
    return page
    
    
def get_random_user_agent() -> tp.Any:
    '''
    Функция создает случайный объект UserAgent для запроса в интернет.
    Return:
        Случайный объект UserAgent
    '''
    user_agent = UserAgent()
    return user_agent.random


def parse_google_search_page(page: Response) -> tp.List[str]:
    '''
    Парсинг ссылок из поиска с HTML-страницы Google
    Parameters:
        page: объект класса requests.models.Response - ответ на requests.get
    Return:
        Список ссылок
    '''
    urls = []
    soup = BeautifulSoup(page.content, 'html.parser')
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http') or href.startswith('https'):
            if not any(bad in href for bad in [
                'google.', 'youtube.', 'accounts.google.com', 'maps.google.com',
                'webcache.googleusercontent.com', 'policies.google.com', 'support.google.com']):
                urls.append(href)  
    return urls


def parse_yandex_search_page(page: Response) -> tp.List[str]:
    '''
    Парсинг ссылок из поиска с HTML-страницы Yandex
    Parameters:
        page: объект класса requests.models.Response - ответ на requests.get
    Return:
        Список ссылок
    '''
    urls = []
    soup = BeautifulSoup(page.content, 'html.parser')
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http') or href.startswith('https'): 
            if not any(bad in href for bad in ['ya.',
            'yandex.', 'market.yandex.ru', 'maps.yandex.ru', 'yastatic.net',
            'passport.yandex.ru', 'yandex.com', 'yandex.by']):
                urls.append(href)
    return urls


def parse_ddg_search_page(page: Response) -> tp.List[str]:
    '''
    Парсинг ссылок из поиска с HTML-страницы DuckDuckGo
    Parameters:
        page: объект класса requests.models.Response - ответ на requests.get
    Return:
        Список ссылок
    '''
    urls = []
    soup = BeautifulSoup(page.content, 'html.parser')
    for link in soup.find_all('a', class_='result__a', href=True):
        href = link['href']
        if href.startswith('http') or href.startswith('https'):
            urls.append(href)
        elif 'uddg=' in href: 
            parsed_url = urllib.parse.urlparse(href)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            decoded_url = query_params.get('uddg', [None])[0]
            if decoded_url:
                urls.append(decoded_url)
    return urls

def parse_rambler_search_page(page: Response) -> tp.List[str]:
    '''
    Парсинг ссылок из XML-ответа Rambler proxy
    Parameters:
        page: объект класса requests.models.Response - ответ на requests.get
    Return:
        Список ссылок
    '''
    
    urls = []
    soup = BeautifulSoup(page.content, features='xml')
    
    main_grouping = soup.find('grouping', attrs={'attr': ''})
    
    if main_grouping and not soup.find('reask'):
        for url_tag in main_grouping.find_all('url'):
            href = url_tag.get_text().strip()
            if href.startswith(('http://', 'https://')):
                urls.append(href)
    return urls

def get_rambler_token():
    token1 = "daf353c7s3hv6a9mnkcx461bsyk"
    token2 = "7sl989dgdj78ewf84lksd8js7bcd"
    return [token1, token2]
