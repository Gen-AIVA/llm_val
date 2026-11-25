# Докстринги тестов в ноутбуке [`5.2.All_Test_example.ipynb`](examples/5.2.All_Test_example.ipynb)

## [`valtest_test_dataset_val`](src/llm_val/val_test/middlegame/valtest_test_dataset_val.py#L43)

`src/llm_val/val_test/middlegame/`

```text
Агрегирующий тест набора данных. Запускает все подтесты из пакета subtests_test_dataset_val и выставляет итоговый светофор.
Parameters:
    sampler: Объект класса Sampler
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    subtests: Список названий подтестов для запуска. Если None, запускаются все найденные подтесты
    exec_env: Среда выполнения ('default' или 'sberds')
    is_info: True, если тест информативный
    subtests_kwargs: Дополнительные параметры отдельных подтестов, где ключ - название подтеста
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_format_check`](src/llm_val/val_test/middlegame/subtests_test_dataset_val/valtest_format_check.py#L59)

`src/llm_val/val_test/middlegame/subtests_test_dataset_val/`

```text
Тест на наличие пропусков в датасете, несоответствий типа колонок и их полного отсутствия
Parameters:
    sampler: Объект класса Sampler
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    optional_columns: Колонки, для которых проверка не требуется
    semaphore_none: Светофорно ли проверяются пропуски в датасете
    is_info: True, если тест информативный
    exec_env: Переменная исполняемого окружения ('default' или 'sberds')
    violation: опицональный результат вычисления индексов значений с типом None (по ключу "NA") и значений c несоответсвущим типом (по ключу "type") в каждой из колонок
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_exact_duplicates`](src/llm_val/val_test/middlegame/subtests_test_dataset_val/valtest_exact_duplicates.py#L49)

`src/llm_val/val_test/middlegame/subtests_test_dataset_val/`

```text
Тест отсутствия дубликатов и внутри выборки.
Если их доля менее threshold (по умолчанию - 10%) - тест пройден.
Parameters:
    sampler: Объект класса Sampler
    data_type: Тип данных, на котором исследуется качество данных
    threshold: Граница светофора для доли дубликатов
    is_info: True, если тест информативный
    data_duplicates_dict: Опциональный словарь с предпосчитанными результатами теста.
        Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
    exec_env: Среда выполнения ('default' или 'sberds')
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_semantic_duplicates`](src/llm_val/val_test/middlegame/subtests_test_dataset_val/valtest_semantic_duplicates.py#L51)

`src/llm_val/val_test/middlegame/subtests_test_dataset_val/`

```text
Тест доли семантических дубликатов вопросов в test выборке.
Parameters:
    sampler: Объект класса Sampler
    exact_dupicates: список кортежей наблюдений-дубликатов
    embedding_client: OpenAI-совместимый клиент с методом embeddings.create. Обязателен при is_long_answer=True
    embedding_model_name: Название эмбеддинг-модели, передаваемое в embeddings.create
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    threshold: Граница светофора - максимально допустимая доля дубликатов в выборке.
    threshold_duplicates: Граница признания семантических дубликатов.
    is_info: True, если тест информативный
    timeout: Максимальное время ожидания одного обращения к модели в секундах
    retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
    max_attempts: Максимальное количество попыток обращения к модели
    exec_env: Переменная исполняемого окружения ('default' или 'sberds')
    embeddings: опицональный результат вычисления нормализованных эмдеддингов запросов
        Если не None, то вычисление не производится, и на основе переданного аргумента считаются статистики.
    similarity: опицональный результат вычисления попарного косинусного расстояния между запросами
        Если не None, то вычисление не производится, и на основе переданного аргумента считаются статистики.
    duplicate_rate: опицональный результат вычисления доли семантических дубликатов.
        Если не None, то вычисление не производится, и на основе переданного аргумента строится вывод теста.
    duplicate_pairs: опицональный результат вычисления списка пар семантических дубликатов.
        Если не None, то вычисление не производится, и на основе переданного аргумента строится вывод теста.
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_train_test_intersections`](src/llm_val/val_test/middlegame/subtests_test_dataset_val/valtest_train_test_intersections.py#L59)

`src/llm_val/val_test/middlegame/subtests_test_dataset_val/`

```text
Тест на отсутствие пересечений между тестовой и валидационной выборками.
Если валидационный датасет не предоставлен, тест не проводится (серый).
Parameters:
    sampler: Объект класса Sampler
    exec_env: Среда выполнения ('default' или 'sberds')
    train_data_type: Тип данных с валидационным датасетом
    test_data_type: Тип данных с тестовым датасетом
    threshold: Граница светофора для доли пересечений
    duplicates: Список групп индексов полных дубликатов в тестовом датасете
    is_info: True, если тест информативный
    precomputed_intersections: Опциональный словарь с предпосчитанными результатами теста.
        Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_outliers_by_char_count`](src/llm_val/val_test/middlegame/subtests_test_dataset_val/valtest_outliers_by_char_count.py#L46)

`src/llm_val/val_test/middlegame/subtests_test_dataset_val/`

```text
Тест отсутствия выбросов среди примеров по количеству символов. 
На основе квартилей Q1, Q3 и межквартильного размаха IQR определяются примеры-выбросы. 
Если их доля менее threshold (по умолчанию - 10%) - тест пройден.
Parameters:
    sampler: Объект класса Sampler
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    threshold: Граница светофора для доли выбросов
    is_info: True, если тест информативный
    input_stats: Словарь со статистиками выбросов по текстам запросов из precomputed
    output_stats: Словарь со статистиками выбросов по текстам ответов (применимо для формата диалогов) из precomputed
    exec_env: Среда выполнения ('default' или 'sberds')
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_dataset_category_proportions`](src/llm_val/val_test/middlegame/subtests_test_dataset_val/valtest_dataset_category_proportions.py#L69)

`src/llm_val/val_test/middlegame/subtests_test_dataset_val/`

```text
Валидационный тест для проверки соответствия фактического распределения категорий в датасете
априорному распределению из отчёта о разработке.
Parameters:
    sampler: Объект класса Sampler с заполненной колонкой 'thema'
    apriori_dict: Словарь с наименованием категории и её априорной долей
    data_type: Тип данных, на которых производится расчёт метрики
    threshold: Порог PSI для светофора (один порог)
    is_info: True, если тест информативный
    exec_env: Среда выполнения ('default' или 'sberds')
    category_stats: Предпосчитанные данные (из precomputed)
Return:
    Словарь в соответствии с концепцией llm_val.
```

## [`valtest_dataset_size_compliance`](src/llm_val/val_test/middlegame/subtests_test_dataset_val/valtest_dataset_size_compliance.py#L48)

`src/llm_val/val_test/middlegame/subtests_test_dataset_val/`

```text
Проверяет, что размер тестового датасета после удаления дубликатов и выбросов
не меньше согласованного минимального объёма.
Parameters:
    sampler: Объект Sampler или словарь (при exec_env='sberds').
    exec_env: 'default' – sampler.test['X'] возвращает DataFrame,
        'sberds' – sampler['test']['X'] возвращает DataFrame.
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    agreed_min_count: Согласованный минимальный объём (из теста 3.3.5).
    outlier_indices: Индексы строк-выбросов, подлежащих удалению.
    duplicate_indices: Индексы строк-дубликатов, подлежащих удалению.
    semantic_duplicate_indices: Индексы строк семантических дубликатов, подлежащих удалению.
    is_info: True, если тест информативный.
    cleaned_count: Предподсчитанное количество примеров после очистки.
    indices_to_remove: Агрегация индексов строк, подлежащих удалению.
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_trace_connectivity`](src/llm_val/val_test/traces/valtest_trace_connectivity.py#L59)

`src/llm_val/val_test/traces/`

```text
Тест на связность трейсов.
Parameters:
    sampler: Объект класса Sampler
    data_type: Ключ, где лежат трейсы агента
    exec_env: среда выполнения ('default' или 'sberds')
    yellow_semaphore_threshold: порог для желтого светофора за все проверки теста
    start_nodes: список начальный вершин графа агента 
    end_nodes: список конечных вершин графа агента
    connected_components: список компонент связности графа
    is_info: True, если тест информаивный 
    **kwargs: дополнительные параметры
Return:
    Словарь в соответствии с концепцией с llm_val
```

## [`valtest_nodes_usage_analysis`](src/llm_val/val_test/traces/valtest_nodes_usage_analysis.py#L55)

`src/llm_val/val_test/traces/`

```text
Тест на анализ использования нод и инструментов условного графа.
Parameters:
    sampler: объект класса Sampler
    data_type: Ключ, где лежат трейсы агента
    exec_env: Среда выполнения ('default' или 'sberds')
    frequency_threshold: минимальный процент трейсов, в которых должна быть задействована каждая нода
    red_semaphore_threshold: верхний порог для красного светофора доли нод, которые встречаются в трейсах меньше, чем frequency_threshold (не включая порог)
    yellow_semaphore_threshold: верхний порог для желтого светофора доли нод, которые встречаются в трейсах меньше, чем frequency_threshold (не включая порог)
    is_info: True, если тест информативный
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_markup_overlap_sufficiency`](src/llm_val/val_test/data_quality/valtest_markup_overlap_sufficiency.py#L46)

`src/llm_val/val_test/data_quality/`

```text
Parameters:
    sampler: Объект класса Sampler или словарь с данными
    data_type: Тип данных, на которых производится расчет метрики.
        Hint: название атрибута в sampler'е
    green_semaphore_threshold: Граница выставления зеленого светофора.
        Минимальное число асессоров для зеленого светофора.
        Имеет наивысший приоритет.
        Если int — единый порог для всех метрик.
        Если dict — порог для каждой метрики отдельно
    n_categories: Количество возможных вариантов оценки (N).
        Порог вычисляется как N+1.
        Если int — единое N для всех метрик.
        Если dict — N для каждой метрики отдельно.
        Если None — N вычисляется автоматически как количество
        уникальных значений метрики в данных
    is_justified: Есть ли обоснование меньшего числа разметчиков,
        чем необходимое перекрытие
    metric_columns: Список названий колонок с метриками.
        Если None — используются все колонки с суффиксом _metric
    exec_env: Среда исполнения ('default' или 'sberds')
    is_info: True, если тест информативный
    overlap_statistics: Предподсчитанная статистика перекрытий
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_annotation_quality_agent`](src/llm_val/val_test/data_quality/valtest_annotation_quality_agent.py#L58)

`src/llm_val/val_test/data_quality/`

```text
Оценка качества разметки на основании согласованности асессоров
Parameters:
    sampler: Объект класса Sampler или словарь с данными
    data_type: Тип данных, на которых производится расчет метрики.
        Hint: название атрибута в sampler'е
    min_periods: Минимальное количество не np.nan элементов
        для подсчета корреляции Спирмена между асессорами
    metric_columns: Список названий колонок с метриками.
        Если None — используются все колонки с суффиксом _metric
    semaphore_threshold_spear: Границы светофора для корреляции Спирмена
    semaphore_threshold_krip: Границы светофора для альфы Криппендорфа
    semaphore_threshold_cohen: Границы светофора для Каппы Коэна
    semaphore_threshold_fleiss: Границы светофора для Каппы Флейса
    exec_env: Среда исполнения ('default' или 'sberds')
    is_info: True, если тест информативный
    criteria_stats: Предподсчитанный список статистик по критериям
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_metric_agent`](src/llm_val/val_test/model_quality/valtest_metric_agent.py#L70)

`src/llm_val/val_test/model_quality/`

```text
Тест на качество прогноза модели по ключевой метрике качества.
Parameters:
    sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
    scorer: Объект класса Scorer. Используется при exec_env='default'
    main_metric: Название ключевой метрики качества.
        Hint: один из ключей словаря, который выдает scorer
    data_type: Тип данных, на которых производится расчет метрики.
        Hint: название атрибута в sampler'е
    metric_agg: Название агрегации метрики.
        Hint: один из ключей в словаре metrics в scorer'е
    semaphore_threshold: Словарь с границами светофора для второстепенных критериев,
        где ключ - название метрики, а значение - кортеж с границами светофора.
        Границы в кортеже указываются в порядке возрастания
    main_semaphore_threshold: Границы светофора для ключевой метрики качества
    red_freq_threshold: Граница по относительной доле красных светофоров
    greater_is_better: True, если чем больше метрика, тем лучше
    exec_env: Среда выполнения. 'default' - стандартная, 'sberds' - SberDS
    calc_metric_fn: Функция подсчета метрики для exec_env='sberds'.
        Сигнатура: (sampler, data_type, metric_agg) -> Dict[str, float]
    is_info: True, если тест информативный
    metric_value: Опциональное значение метрики.
        Если не None, то подсчет метрики не производится,
        и на основе переданного аргумента выставляется светофор
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_stat_significance`](src/llm_val/val_test/model_quality/valtest_stat_significance.py#L48)

`src/llm_val/val_test/model_quality/`

```text
Тест на статзначимость различия ключевой метрики качества
Parameters:
    sampler: Объект класса Sampler
    scorer: Объект класса Scorer. Используется при exec_env='default'
    metric_agg: Название агрегации метрики.
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    value_threshold: Пороговое значение ключевой метрики качества при использовании 1-выбор-ого крит-ия Стьюдента
    extra_samplers: Дополнительные объекты Sampler, используемые, если в тесте участвует более одной модели
    same_questions: True, если разные модели отвечали на одинаковые вопросы; False, если на разные
    semaphore_threshold: Границы светофора
    bootstrap: Флаг для использования bootstrap
    bootstrap_iterations: Число bootstrap итераций
    bootstrap_ci_type: Тип доверительного интервала ('percentile', 'basic', 'normal')
    alternative: Тип альтернативной гипотезы ('greater', 'less', 'two-sided')
    exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
    random_state: seed генератора случайных чисел
    is_info: True, если тест информативный
    p_value: Опциональное значение p-value.
        Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
    calc_metric_fn: Пользовательская функция расчета метрики.
        Используется при metric_agg != 'single_mean' и scorer is None.
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_metric_by_trivial_params`](src/llm_val/val_test/model_quality/valtest_metric_by_trivial_params.py#L72)

`src/llm_val/val_test/model_quality/`

```text
Тест на анализ динамики ключевой метрики качества модели в зависимости от простых статистик
Parameters:
    sampler: Объект класса Sampler
    scorer: Объект класса Scorer
    embedding_client: OpenAI-совместимый клиент с методом embeddings.create. Обязателен при is_long_answer=True
    embedding_model_name: Название эмбеддинг-модели, передаваемое в embeddings.create
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    business_threshold: Минимальный проходной порог по КМ (при наличии). Необходим для выставления красного светофора
    context_test: флаг проведения проверки для контекста
    token_counter_func: Функция подсчета токенов, List[str] -> np.ndarray[int]
        Принимает тексты запросов и возвращает их  длины в токанех
        Не рекомендуется передавать при наличии колонки llm_prompt_tokens в трейсах, так как токенизация с помощью данной функции будет иметь высший приоритет
    clustering_algo: Алгоритм кластеризации ('kmeans' или 'hdbscan'). Параметры выбираются по умолчанию, подробнее можно посмотреть в соотвествующих модулях
    bins: Название способа бинаризации как параметр функции np.histogram_bin_edges или количество бинов
    metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
    stattest_method: Название статтеста
                     - 'welch' - t-тест Уэлча
                     - 'MW' - U-тест Манна-Уитни
                     - 'bootstrap' - bootstrap
        Метод будет автоматически изменен на бутстрап, если формат метрики не подходит под условия статтеста
    n_bootstrap: Количество итераций бутстрапирования
    min_count_size: Минимальное количество семплов в бине или кластере для корректирования светофора
    min_count_pct: Минимальное доля семплов в бине или кластере для корректирования светофора
        Количество семплов в бинах в хвостах распределения по длинам будет корректироваться на основании этих двух значений
    semaphore_threshold_ratio: Граница светофора для относительного падения качества (1 - <значение указанное в тексе методики>)
    semaphore_threshold_p_val: Граница светофора для уровня значимости падения качества
    random_state: Рандом стейт
    greater_is_better: True, если чем больше метрика, тем лучше
    is_info: True, если тест информативный
    timeout: Максимальное время ожидания одного обращения к модели в секундах
    retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
    max_attempts: Максимальное количество попыток обращения к модели
    exec_env: Переменная исполняемого окружения ('default' или 'sberds')
    len_questions: опицональный результат подсчета длин запросов
        Если не None, то вычисление не производится, и на основе переданного аргумента строятся графики. И если est_by_len_result=None, то также считаются статистики.
    embeddings: опицональный результат вычисления эмдеддингов запросов
        Если не None и est_by_cluster_result=None, то вычисление не производится, и на основе переданного аргумента считаются статистики.
    len_context: опицональный результат подсчета длин контекста
        Если не None, то вычисление не производится, и на основе переданного аргумента строятся графики. И если est_by_context_result=None, то также считаются статистики.
    est_by_len_result: опицональный результат теста для зависимости корректности запроса от его длины
        Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
    est_by_cluster_result: опицональный результат теста для зависимости корректности запроса от его кластера
        Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
    est_by_context_result: опицональный результат теста для зависимости корректности запроса от длины контекста
        Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_loop_duration_analysis`](src/llm_val/val_test/traces/valtest_loop_duration_analysis.py#L26)

`src/llm_val/val_test/traces/`

```text
Тест на оценку времени работы и степени зацикленности агента.
Parameters:
    sampler: Объект класса Sampler
    data_type: Ключ, где лежат трейсы агента
    exec_env: среда выполнения ('default' или 'sberds')
    loop_time_stats: датафрейм с рассчитанными статистиками по времени работы и мере зацикленности графа агента
Return:
    Словарь в соответствии с концепцией с llm_val
```

## [`valtest_criterion_informativeness`](src/llm_val/val_test/data_quality/valtest_criterion_informativeness.py#L71)

`src/llm_val/val_test/data_quality/`

```text
Для каждого критерия вычисляется мера информативности на основе нормализованных
оценок асессоров. Критерий считается неинформативным, если мера ниже порога.
Parameters:
    sampler: объект Sampler или словарь с данными в зависимости от exec_env
    data_type: Тип данных, на которых производится расчет метрики.
        Hint: название атрибута в sampler'е
    criterion_columns: список колонок с критериями.
        Если None, используются колонки, заканчивающиеся на '_metric'.
    uninformativeness_threshold: пороговое значение меры (ниже которого критерий неинформативен)
    measure_func: функция для вычисления меры информативности.
        Сигнатура: (values: List[float]) -> float.
        По умолчанию – дисперсия нормализованных значений.
    semaphore_threshold: границы светофора для доли неинформативных критериев
    is_info: если True, светофор всегда 'gray'
    exec_env: окружение выполнения ('default' или 'sberds')
    measures: предварительно вычисленные значения мер.
        Если не None, пересчёт не выполняется.
Returns:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_metric_by_thema_agent`](src/llm_val/val_test/model_quality/valtest_metric_by_thema_agent.py#L61)

`src/llm_val/val_test/model_quality/`

```text
Тест достижения качества по ключевой метрике в разрезе тематик/доменов
Parameters:
    sampler: Объект класса Sampler или словарь
    scorer: Объект класса Scorer
    main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
    business_metric_threshold: Бизнес-критерий для ключевой метрики качества
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
    significance_level: Уровень значимости для проверки бизнес-критерия
    main_column: Название колонки в y с объектным значением ключевой метрики
    greater_is_better: True, если чем больше метрика, тем лучше
    other_scenarios: Сценарии, не учитываемые в финальном светофоре
    exec_env: Среда выполнения. Принимает значения 'default' или 'sberds'
    bootstrap: Флаг для использования bootstrap при расчете p-value
    bootstrap_iterations: Число bootstrap итераций
    random_state: seed генератора случайных чисел
    metric_fn: Функция агрегации объектных значений метрики для bootstrap
    is_info: True, если тест информативный
    metric_value_by_scenario: Опциональное значение метрики.
        Если не None, то подсчет метрики не производится, и на основе переданного аргумента выставляется светофор.
    p_value_by_scenario: Опциональные значения p-value метрик в разрезе сценариев.
        Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_metric_by_dialog_len`](src/llm_val/val_test/model_quality/valtest_metric_by_dialog_len.py#L49)

`src/llm_val/val_test/model_quality/`

```text
Тест динамики ключевой метрики в зависимости от длины истории диалога: сравнивает среднюю КМ на 10% самых длинных диалогов со средней КМ на остальных диалогах с помощью одностороннего двухвыборочного t-теста Уэлча отдельно по длине в токенах и по числу реплик.
Parameters:
    sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    main_column: Название колонки с ключевой метрикой качества в y; если None, усредняются все колонки y
    significance_level: Уровень значимости для выставления светофора
    mean_decrease_threshold: Относительная граница падения средней КМ на подвыборке
    greater_is_better: True, если чем больше метрика, тем лучше
    is_info: True, если тест информативный
    stats_traces: Опциональные статистики проверки по токенам из precomputed
    stats_dialogue: Опциональные статистики проверки по репликам из precomputed
    exec_env: Среда выполнения ('default' или 'sberds')
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_answer_reprod_agent`](src/llm_val/val_test/stability_without_target/valtest_answer_reprod_agent.py#L63)

`src/llm_val/val_test/stability_without_target/`

```text
Тест стабильности воспроизведения ответов агента на схожих запросах: оценивает смысловую и языковую стабильность ответов решения путем многократного реинференса отдельных запросов.
Parameters:
    sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    model: Валидируемая языковая модель, имеет метод predict(запросы)
    is_long_answer: True, если ответы развернутые, иначе ожидаются короткие ответы
    is_creative: True для творческой задачи, False для точной
    n_generated_answers: Число повторных генераций ответов
    n_gramms_length: Длина n-грамм для лексического разнообразия
    embedding_client: OpenAI-совместимый клиент с методом embeddings.create. Обязателен при is_long_answer=True
    embedding_model_name: Название эмбеддинг-модели, передаваемое в embeddings.create
    use_augmentations: Флаг включения проверки на аугментированных данных
    augmentations_list: Список символьных аугментаций: 'use_deletion', 'use_swap', 'use_insert', 'use_bf',
        'use_special_symbol', 'use_translit_symbol'. По умолчанию None,
        тогда используется ['use_special_symbol'] для выполнения шага аугментаций из методики без явного указания
    p_augmented: Вероятность аугментации символа
    diversity_threshold_pp: Нижний и верхний пороги разнообразия в долях
    rel_threshold: Порог относительной разницы
    seed: Начальное значение датчика случайных чисел
    timeout: Максимальное время ожидания одного обращения к модели в секундах
    retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
    max_attempts: Максимальное количество попыток обращения к модели
    exec_env: Среда выполнения ('default' или 'sberds')
    is_info: True, если тест информативный
    orig_diversity: Опциональное разнообразие на исходных данных из precomputed
    aug_diversity: Опциональное разнообразие на аугментированных данных из precomputed
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_metric_aug_stability_word`](src/llm_val/val_test/stability/valtest_metric_aug_stability_word.py#L99)

`src/llm_val/val_test/stability/`

```text
Тест на стабильность прогноза модели в зависимости от словесных аугментаций с расчетом статистической значимости.
Поддерживает работу без передачи model/aug_client, если в данных уже есть готовые aug- и pred-колонки
с корректным неймингом (aug_valtest_metric_aug_stability_word_* и pred_aug_valtest_metric_aug_stability_word_*).
Parameters:
    sampler: Объект класса Sampler
    scorer: Объект класса Scorer
    main_metric: Название ключевой метрики качества
    model: Валидируемая модель; имеет метод predict.
    augmentations_list: Список выполняемых аугментаций
    translation_language: Язык для аугментации translation (указанный на русском языке)
    word_replace_ratio: Доля слов для замены при синонимической аугментации
    answer_checker: Функция автооценки ответа с сигнатурой (запрос, ответ модели, эталонный ответ)
    metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
    greater_is_better: True, если чем больше метрика, тем лучше
    is_info: True, если тест информативный
    seed: Начальное значение датчика случайных чисел
    max_attempts: Максимальное количество попыток аугментации для текстов, которые не удалось изменить
    timeout: Максимальное время ожидания одного вызова model.predict в секундах
    retry_delay: Базовая задержка между повторными попытками вызова model.predict в секундах
    exec_env: Среда выполнения ('default' или 'sberds')
    aug_client: OpenAI-совместимый клиент для генерации аугментаций.
    aug_model_name: Название модели для генерации аугментаций
    aug_model_prompts: Опциональный словарь с системными промптами для аугментаций.
                       Допустимые ключи: 'paraphrase', 'synonym', 'translation_fwd', 'translation_bwd'.
                       Если ключ отсутствует или словарь None, используются промпты по умолчанию.
    baseline_per_aug: Опциональный словарь с базовыми метриками
    metrics_augmented: Опциональный словарь с метриками на аугментированных данных
    p_values: Опциональный словарь с рассчитанными p-values
Return:
    Словарь в соответствии с концепцией llm_val, содержащий отчет и предвычисленные значения
```

## [`valtest_metric_aug_stability_symbol`](src/llm_val/val_test/stability/valtest_metric_aug_stability_symbol.py#L56)

`src/llm_val/val_test/stability/`

```text
Тест на стабильность прогноза модели в зависимости от символьных аугментаций запросов
Parameters:
    sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
    scorer: Объект класса Scorer
    model: Валидируемая модель; имеет метод predict(запросы)
    main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
    augmentations_list: Список выполняемых аугментаций 
        доступны: 'human_level_augmentations', 'symbol_swap', 'butter_finger'.
    p_augmented: Вероятность аугментации символа каждой аугментацией
    answer_checker: Функция автооценки ответа с сигнатурой (запрос, ответ модели, эталонный ответ)
    metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
    greater_is_better: True, если чем больше метрика, тем лучше
    seed: Начальное значение датчика случайных чисел
    timeout: Максимальное время ожидания одного обращения к модели в секундах
    retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
    max_attempts: Максимальное количество попыток обращения к модели
    exec_env: Среда выполнения ('default' или 'sberds')
    is_info: True, если тест информативный
    metric_baseline: Опциональный словарь со значением метрики на данных без изменений из precomputed
    metrics_augmented: Опциональный словарь со словарями значений метрики на аугментированных данных из precomputed
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_rag_top_stability_agent`](src/llm_val/val_test/stability_without_target/valtest_rag_top_stability_agent.py#L67)

`src/llm_val/val_test/stability_without_target/`

```text
Тест на стабильность выдачи RAG в зависимости от аугментаций текстов на уровне символов или слов

Parameters:
    sampler: Объект класса Sampler
    executor: Объект класса RAGExecutor
    augmentations_list: Список выполняемых аугментаций.
        Поддерживаемые значения: 'symbol_swap', 'butter_finger', 'human_level_augmentations' (символьные),
        'paraphrase', 'synonym', 'translation' (словные).
    exec_env: Среда выполнения ('default' или 'sberds')
    data_type: Тип данных, на которых производится расчет метрики (название атрибута в sampler)
    semaphore_threshold: Красный и желтый порог светофора
    p_augmented: Вероятность аугментации символа каждой аугментацией
    translation_language: Язык для аугментации translation (указанный на русском языке)
    word_replace_ratio: Доля слов для замены при синонимической аугментации
    seed: Начальное значение датчика случайных чисел
    max_attempts: Максимальное количество попыток аугментации для текстов, которые не удалось изменить
    aug_client: OpenAI-совместимый клиент для генерации аугментаций
    aug_model_name: Название модели для генерации аугментаций
    aug_model_prompts: Опциональный словарь с системными промптами для аугментаций.
                       Допустимые ключи: 'paraphrase', 'synonym', 'translation_fwd', 'translation_bwd'.
                       Если ключ отсутствует или словарь None, используются промпты по умолчанию.
    is_info: True, если тест информативный
    iou_symbol_results: Опциональный словарь с предпосчитанными результатами теста на уровне символов.
        Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
    iou_word_results: Опциональный словарь с предпосчитанными результатами теста на уровне слов.
        Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_rag_quality_agent`](src/llm_val/val_test/rag_quality/valtest_rag_quality_agent.py#L69)

`src/llm_val/val_test/rag_quality/`

```text
Тест на анализ качества RAG-системы.
Parameters:
    sampler: Объект класса Sampler.
    data_type: Тип данных, на которых производится расчет метрики.
    topk: Количество элементов для расчета метрик.
    metric_list: Cписок метрик для расчета. Hint: ключи словаря eval_function_ranking в модуле llm_val.utils.metrics_rag.
    thresholds: Словарь с порогами для передаваемых метрик;
        Ключ - название метрики, значение - кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора.
    default_thresholds: Кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора;
        Используется как дефолтное значение в случае отсутсвия порогов в thresholds для метрик из metric_list.
    is_info: True, если тест информативный.
    avg_metrics: Опциональный словарь с предпосчитанными результатами теста.
        Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
    exec_env: Среда выполнения ('default' или 'sberds')
Return:
    Словарь в соответствии с концепцией llm_val.
```

## [`valtest_reflection_usefulness`](src/llm_val/val_test/reflection/valtest_reflection_usefulness.py#L29)

`src/llm_val/val_test/reflection/`

```text
Тест на оценку полезности рефлексии путем сравнения косинусного расстояния между ответами до и после рефлексии.
Parameters:
    sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
    exec_env: Среда выполнения ("default" или "sberds")
    reflection_nodes: Опциональный список названий нод рефлексии (span_name), для которых нужно оценить полезность.
                      Должен быть передан, если avg_distances не передан.
    embedding_client: OpenAI-совместимый клиент с методом embeddings.create. Должен быть передан, если avg_distances не передан.
    embedding_model_name: Название эмбеддинг-модели, передаваемое в embeddings.create
    data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    timeout: Максимальное время ожидания одного обращения к модели в секундах
    retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
    max_attempts: Максимальное количество попыток обращения к модели
    avg_distances: Опциональный список готовых средних косинусных расстояний.
           Формат: [(название_ноды, среднее_расстояние), ...]
           Если передан, расчеты не выполняются.
Return:
    Словарь в соответствии с концепцией llm_val
```

## [`valtest_tool_metric_dynamics`](src/llm_val/val_test/traces/valtest_tool_metric_dynamics.py#L33)

`src/llm_val/val_test/traces/`

```text
Определяет поведение ключевой метрики в зависимости от инструментов.

Тест работает только с трейсами, имеющими в колонке aef_kind
значение tool. Инструменты, присутствующие во всех сессиях, удаляются.

Parameters:
    sampler: Объект класса Sampler с трейсами
    scorer: Объект класса Scorer для подсчёта метрики качества.
        Если None — метрика считается через calc_metric_fn (или calc_metric_from_data)
    data_type: Тип данных, на которых производится расчёт.
        Hint: название атрибута в sampler'е
    metric_column: Название колонки в sampler[data_type]['y'],
        содержащей целевую метрику качества
    metric_agg: Название агрегации метрики.
        Hint: один из ключей в словаре metrics в scorer'е
    greater_is_better: True, если чем больше метрика, тем лучше
    exec_env: Среда выполнения ('default' или 'sberds')
    calc_metric_fn: Функция подсчёта метрики при scorer=None.
        Сигнатура: (data: pd.DataFrame, metric_agg: str) -> Dict[str, float].
        Если None — используется calc_metric_from_data
    precomputed_report: Предподсчитанная таблица результатов.
        Если передана, расчёт не производится
Return:
    report в соответствии с концепцией llm_val
```

## [`valtest_incorrect_tool_calls`](src/llm_val/val_test/traces/valtest_incorrect_tool_calls.py#L115)

`src/llm_val/val_test/traces/`

```text
Тест на проверку корректности вызова функций (инструментов) агентом.

Читает трейсы AEF-формата, отфильтровывает спаны с aef_kind='tool',
разбирает аргументы из input_text и, при однозначном совпадении
с сигнатурой, извлекает вложенный словарь фактических аргументов
и вычисляет долю некорректных вызовов — как по каждому инструменту,
так и в целом.

Дополнительно (п.3 методики): если переданы OpenAI-compatible клиент
и model_name, тест также семантически проверяет аргументы
вызова относительно пользовательского запроса. п.3 не влияет на
итоговый светофор и является информативной метрикой.

Parameters:
    sampler: объект класса Sampler
    data_type: ключ, где лежат трейсы агента
    judge_client: OpenAI-compatible клиент для сравнения фактических аргументов с требуемыми в p3
    judge_model_name: Название модели судьи
    check_p3: запускать семантическую проверку P3. По умолчанию True.
        P3 выполняется, только если также переданы judge_client и judge_model_name.
    system_prompt: системный промт для судьи
        Если параметр - None, то используется промт по умолчанию.
        Основные рекомендации при составлении промта:
        На вход судье передаются пользовательский запрос, история диалога при её наличии, имя инструмента и сериализованные аргументы вызова
        Судья оценивает только то, позволяют ли аргументы выполнить намерение пользователя
        Судья не проверяет наличие инструмента в реестре, сигнатуру, типы значений, лишние параметры и JSON-валидность
        Если в контексте недостаточно сведений для выполнения запроса, вердикт — INCORRECT
        Если аргументы включают достаточные конкретизирующие данные либо допустимое значение по умолчанию, вердикт — CORRECT
        Итоговый формат ответа строго ограничен одним словом: CORRECT или INCORRECT
    temperature: температура модели судьи
    max_tokens: максимальное количество токенов в ответе судьи
    exec_env: среда выполнения ('default' или 'sberds')
    signatures: словарь {имя_функции: {параметр: тип}} с эталонными сигнатурами
    tool_descriptions: Опциональные человекочитаемые описания назначения
        инструментов в формате {tool_name: description}. Используются
        только для формирования промта судьи для проверки в п.3.
    timeout: Максимальное время ожидания одного обращения к модели в секундах
    retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
    max_attempts: Максимальное количество попыток обращения к модели
    is_info: True, если тест информативный
    per_tool_stats: предвычисленная статистика по инструментам (опционально)
    red_total_rate_threshold: порог общей доли ошибок для красного светофора
    yellow_per_tool_threshold: порог доли ошибок для отдельного инструмента для жёлтого светофора
Return:
    Словарь в соответствии с концепцией llm_val.
```
