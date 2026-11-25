import re
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema
from pandera.dtypes import String


OFFICIAL_VALUE_MAP = {
    "ML-задача": {
        "domain knowledge": "Domain knowledge",
        "summarization": "Summarization",
        "generation": "Generation",
        "chat": "Chat",
        "classification": "Classification",
        "extract": "Extract",
        "brainstorming": "Brainstorming",
        "classification + extract": "Classification + Extract",
        "question answering": "Question Answering",
        "question answering + rag": "Question Answering + RAG",
        "rewrite / editing": "Rewrite / Editing",
        "text translation": "Text translation",
        "core quality": "Core Quality",
    },
    "Уровень знаний": {
        "l0": "L0",
        "l1": "L1",
        "l2": "L2",
    },
    "Метод оценки качества": {
        "асессоры": "Асессоры",
        "авто": "Авто",
        "автометрика": "Автометрика",
        "llm-as-judge": "LLM-as-judge",
    },
    "Наименование ключевой метрики": {
        "accuracy": "ACCURACY",
        "mape": "MAPE",
        "rouge": "Rouge",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
    },
    "Домен знаний": {
        "документы": "Документы",
    "продукты сбера": "Продукты Сбера",
    "право": "Право",
    "базы данных/знаний": "Базы данных/знаний",
    "компьютерные науки": "Компьютерные науки",
    "кибербезопасность": "Кибербезопасность",
    "код": "Код",
    "финансы": "Финансы",
    "биология": "Биология",
    "математика": "Математика",
    "разное": "Разное",
    "сельское хозяйство": "Сельское хозяйство",
    "образование": "Образование",
    "бизнес": "Бизнес",
    "химия": "Химия",
    "экономика": "Экономика",
    "география": "География",
    "история": "История",
    "медицина": "Медицина",
    "философия": "Философия",
    "этика": "Этика",
    "физика": "Физика",
    "психология": "Психология",
    "обществознание": "Обществознание",
    "логика": "Логика",
    "политика": "Политика",
    "кино": "Кино",
    "литература": "Литература",
    "музыка": "Музыка",
    "недвижимость": "Недвижимость",
    "нефтегазовая промышленность": "Нефтегазовая промышленность",
    "связь и телекоммуникации": "Связь и телекоммуникации",
    "транспорт и логистика": "Транспорт и логистика",
    "энергетика": "Энергетика",
    "туризм": "Туризм",
    "агропромышленный комплекс": "Агропромышленный комплекс",
    "лесопромышленный комплекс": "Лесопромышленный комплекс",
    "жилая недвижимость": "Жилая недвижимость",
    "коммерческая недвижимость": "Коммерческая недвижимость",
    "металлургия": "Металлургия",
    "рыбодобывающее производство": "Рыбодобывающее производство",
    "строительные подрядчики": "Строительные подрядчики",
    "торговля": "Торговля",
    "услуги": "Услуги",
    "ассистент": "Ассистент",
    },
    "Блок/ДЗО": {
        "b2c": "B2C",
        "grc": "GRC",
        "бсп": "БСП",
        "бср": "БСР",
        "дзо": "ДЗО",
        "киб": "КИБ",
        "лик": "ЛиК",
        "не сбер": "Не Сбер",
        "сервисы": "Сервисы",
        "технологии": "Технологии",
    },
}


ALLOWED_GIGACHAT_MARKETING = [
    "GigaChat",
    "GigaChat-Max",
    "GigaChat-Pro",
    "GigaChat-Plus",
    "GigaChat-2",
    "GigaChat-2-Max",
    "GigaChat-2-Pro",
    "GigaChat-2-Reasoning",
    "GigaChat-Lite",
    "GigaChat-7b +  LoRA",
    "GigaChat-7b.3.1-24.3-32k (SFT)",
    "GigaChat-3-Ultra"
]


ALLOWED_GIGACHAT_MODEL_NAMES = [
    "GigaChat-90b-32k-base:1.0.26.20",
    "GigaChat-20b-32k-base:1.0.26.20",
    "GigaChat-20b-32k-base.0.26.15",
    "GigaChat-30b-32k-base.0.26.8",
    "GigaChat-30b-8k-base.0.26.15",
    "GigaChat-75b-32k-base.0.26.20",
    "GigaChat-7b-32k-base-4.0.26.8",
    "GigaChat-7b-32k-function-2.2.25.3",
    "GigaChat-7b-8k-function-2.2.25.3",
    "GigaChat-90b-32k-base:1.0.27.04",
    "GigaChat-75b-32k-base.0.27.04",
    "GigaChat-20b-32k-base:1.0.27.04",
    "GigaChat-20b-32k-base:2.0.27.04",
    "GigaChat-90b-32k-base:2.0.27.04",
    "GigaChat-75b-32k-base-2.0.27.04",
    "GigaChat-75b-32k-base-2.0.28.02",
    "GigaChat-20b-32k-base:2.0.28.02",
    "GigaChat-90b-32k-base:2.0.28.02",
    "GigaChat-2-Max:2.0.30.00",
    "GigaChat-2-Pro:2.0.30.00",
    "GigaChat-2:2.0.30.00",
    "GigaChat-2-Max:2.0.30.01",
    "GigaChat-2-Pro:2.0.30.01",
    "GigaChat-2:2.0.30.01",
    "GigaChat-3.1.25.3-8b-7k",
    "GigaChat-7b-4k-base.0.22.4",
    "GigaChat-29b-8k-base-3.2.24.3",
    "GigaChat-29b-8k-function-2.2.25.3",
    "GigaChat-75b-32k-base:1.0.26.20",
    "GigaChat-75b-32k-base:2.0.30.01",
    "GigaChat-75b-32k-base:2.0.28.02",
    "GigaChat-7b-8k-base-3.2.24.3",
    "GigaChat-90b-32k-function.0.26.20",
    "GigaChat-2:2.0.28.2",
    "GigaChat-Pro-2:2.0.28.2",
    "GigaChat-Max-2:2.0.28.2",
    "GigaChat-20b-32k-function.0.26.20",
    "GigaChat-7b-32k-function-4.0.26.8",
    "GigaChat-75b-32k-function.0.26.20",
    "GigaChat-20b-32k-base.0.26.20",
    "GigaChat-7b.3.1-24.3-32k (SFT)",
    "GigaChat3b-2k-base.0.19.3",
    "GigaChat-20b-32k-base:1.0.26.15",
    "GigaChat-30b-32k-base:1.0.26.15",
    "GigaChat-7b-32k-base:4.0.26.8",
    "GigaChat-30b-8k-base:1.0.26.8",
    "GigaChat:2.1.28.2",
    "GigaChat-2-Reasoning:2.0.29.05",
    "GigaChat-30b-8k-base.0.26.8",
    "GigaChat-30b-32k-base.0.26.15",
    "GigaChat-7b +  LoRA",
    "GigaChat-3-Ultra:32.1.15.1"
    ]


def apply_official_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, mapping in OFFICIAL_VALUE_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)

    return df


def normalize_for_isin(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": pd.NA})
    )


def normalize_dataset_name(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(r"\.(jsonl|csv|parquet|txt)$", "", regex=True)
        .replace({"nan": pd.NA})
    )


def check_business_criterion(df: pd.DataFrame) -> bool:
    crit = df["Бизнес-критерий использования модели в пром"]
    stat = df["Стат-значимо ли выполнен бизнес критерий?"]

    problem = crit.notna() & stat.isna()
    return not problem.any()


def check_sut_for_L1(df: pd.DataFrame) -> bool:
    mask = df["Уровень знаний"].astype(str) == "l1"
    wrong = df.loc[mask, "Суть бизнес-задачи"] != "Проверка доменных знаний"
    return not wrong.any()


def check_id_version_requirement(df: pd.DataFrame) -> bool:
    if "id версии модели по БМ" not in df.columns:
        return True

    mask_L2 = df["Уровень знаний"].astype(str) == "l2"

    wrong = df.loc[mask_L2, "id версии модели по БМ"].isna()
    return not wrong.any()


def check_domain_nullable_only_for_L0(df: pd.DataFrame) -> bool:
    if "Домен знаний" not in df.columns or "Уровень знаний" not in df.columns:
        return True

    mask_invalid = (
        df["Домен знаний"].isna()
        & df["Уровень знаний"].notna()
        & (df["Уровень знаний"].str.upper() != "L0")
    )

    return not mask_invalid.any()


def check_confidence_interval_for_accuracy(df: pd.DataFrame) -> bool:

    if "Наименование ключевой метрики" not in df.columns or "Доверительный интервал метрики качества " not in df.columns:
        return True

    accuracy_mask = df["Наименование ключевой метрики"] == "ACCURACY"

    if not accuracy_mask.any():
        return True

    ci_series = df.loc[accuracy_mask, "Доверительный интервал метрики качества "]

    if ci_series.isna().any():
        return False

    pattern = re.compile(
        r"^\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)$"
    )

    return ci_series.astype(str).apply(lambda x: bool(pattern.match(x))).all()


def check_marketing_model_name(df: pd.DataFrame) -> bool:
    col = "Маркетинговое наименование модели"
    if col not in df.columns:
        return True

    series = df[col].dropna().astype(str)

    mask = series.str.startswith("GigaChat")
    invalid = series[mask].isin(ALLOWED_GIGACHAT_MARKETING) == False

    return not invalid.any()


def check_model_name(df: pd.DataFrame) -> bool:
    col = "Наименование модели"
    if col not in df.columns:
        return True

    series = df[col].dropna().astype(str)

    mask = series.str.startswith("GigaChat")
    invalid = series[mask].isin(ALLOWED_GIGACHAT_MODEL_NAMES) == False

    return not invalid.any()


schema = DataFrameSchema(
    {
        "ML-задача": Column(
        str,
        parsers=pa.Parser(normalize_for_isin),
    checks=[
        Check.isin(
            [
                "domain knowledge", "summarization", "generation", "chat",
                "classification", "extract", "brainstorming",
                "classification + extract", "question answering",
                "question answering + rag", "rewrite / editing",
                "text translation", "core quality"
            ],
                error="Допустимые значения: 'Domain knowledge', 'Summarization', 'Generation', 'Chat', 'Classification', 'Extract', 'Brainstorming', 'Classification + Extract', 'Question Answering', 'Question Answering + RAG', 'Rewrite / Editing', 'Text translation', 'Core Quality'"
            )
        ],
        nullable=False,
        description="Тип ML-задачи",
        title="ML задача",
        coerce=True
    ),
    "Уровень знаний": Column(
        str,
        parsers=pa.Parser(normalize_for_isin),
        checks=[
            Check.isin(
                ["l0", "l1", "l2"],
                error="Допустимые уровни знаний: 'L0', 'L1', 'L2'"
            )
        ],
        nullable=False,
        description="Уровень знаний модели",
        title="Уровень знаний",
        coerce=True
    ),
    "Датасет (jira)": Column(
        str,
        parsers=pa.Parser(lambda x: x.str.strip()),
        checks=[
            Check.str_matches(
                r"^https://jira\.sberbank\.ru/browse/(GCMARKUP|GIGAUSAGE)-\d+$",
                error="Ссылка должна быть в формате: https://jira.sberbank.ru/browse/GCMARKUP-XXX или GIGAUSAGE-XXX"
            )
        ],
        nullable=True,
        description="Ссылка на датасет в Jira",
        title="Jira ссылка",
        coerce=True
    ),
    
    "Метод оценки качества": Column(
        str,
        parsers=pa.Parser(normalize_for_isin),
        checks=[
            Check.isin(
                ["асессоры", "авто", "автометрика", "llm-as-judge"],
                error="Допустимые методы: 'Асессоры', 'Авто', 'Автометрика', 'LLM-as-judge'"
            )
        ],
        nullable=True,
        description="Метод оценки качества",
        title="Метод оценки",
        coerce=True
    ),
    "Наименование ключевой метрики": Column(
        str,
        parsers=pa.Parser(normalize_for_isin),
        nullable=False,
        description="Ключевая метрика качества",
        title="Ключевая метрика",
        coerce=True
    ),
    "Маркетинговое наименование модели": Column(
        str,
        nullable=False
    ),
    "Качество модели": Column(
    float,
    nullable=False,
    coerce=True
    ),
    "Размер датасета": Column(
        nullable=False
    ),
    "Наименование датасета в OBM": Column(
    str,
    parsers=pa.Parser(normalize_dataset_name),
    nullable=True
    ),
    "Домен знаний": Column(
        str,
        parsers=pa.Parser(normalize_for_isin),
        checks=[
            Check.isin(
                list(OFFICIAL_VALUE_MAP["Домен знаний"].keys()),
                error="Допустимые значения: 'Документы', 'Продукты Сбера', 'Право', 'Базы данных/знаний', 'Компьютерные науки', 'Кибербезопасность', 'Код', 'Финансы', 'Биология', 'Математика', 'Разное', 'Сельское хозяйство', 'Образование', 'Бизнес', 'Химия', 'Экономика', 'География', 'История', 'Медицина', 'Философия', 'Этика', 'Физика', 'Психология', 'Обществознание', 'Логика', 'Политика', 'Кино', 'Литература', 'Музыка', 'Недвижимость', 'Нефтегазовая промышленность', 'Связь и телекоммуникации', 'Транспорт и логистика', 'Энергетика', 'Туризм', 'Агропромышленный комплекс', 'Лесопромышленный комплекс', 'Жилая недвижимость', 'Коммерческая недвижимость', 'Металлургия', 'Рыбодобывающее производство', 'Строительные подрядчики', 'Торговля', 'Услуги', 'Ассистент'"
            )
        ],
        nullable=True
    ),
    "Тематика":Column(
        str,
        nullable=True
    ),
    "Суть бизнес-задачи": Column(
        str,
        nullable=False
    ),
    "Наименование задачи": Column(
        str,
        nullable=False
    ),
    "Бизнес-критерий использования модели в пром": Column(
    float,
    nullable=True,
    coerce=True
    ),
    "Команда разработки": Column(
        str,
        nullable=True
    ),
    "Блок/ДЗО": Column(
        str,
        parsers=pa.Parser(normalize_for_isin),
        checks=[
            Check.isin(
                ['b2c', 'grc', 'бсп', 'бср', 'дзо', 'киб', 'лик', 'не сбер', 'сервисы', 'технологии'],
                error="Допустимые значения: 'B2C', 'GRC', 'БСП', 'БСР', 'ДЗО', 'КИБ', 'ЛиК', 'Не Сбер', 'Сервисы', 'Технологии'"
            )
        ],
        nullable=False
    ),

    "Дата прогона": Column(
        nullable=False
    ),
    "Наименование модели": Column(
        str,
        nullable=False
    ),
    "Стат-значимо ли выполнен бизнес критерий?": Column(
        nullable=True
    ),

    "Особенности архитектуры решения": Column(
        str,
        nullable=True
    ),
    "Количество выделенных ПШЕ на SteerCo": Column(
        nullable=True
    ),
    "Ожидаемый фин. эффект. (млн. руб./год) всей инициативы": Column(
        nullable=True
    ),
    "Подтвержденный фин. эффект. (млн. руб./год) всей инициативы": Column(
        nullable=True
    ),
    "Доверительный интервал метрики качества ": Column(
        nullable=False
    ),
    "User ID": Column(
        str,
        parsers=pa.Parser(lambda x: x.str.strip()),
        checks=[
            Check.str_matches(
                r"^CI\d+$",
                error="User ID должен быть вида CI********"
            )
        ],
        nullable=True,
        coerce=True
    ),
    "Наименование инициативы Фин. Деп.": Column(
        str,
        nullable=True
    )
    },

    checks=[
        Check(check_business_criterion, 
              error="Если заполнен Бизнес-критерий использования модели в пром, должно быть заполнено Стат-значимо ли выполнен бизнес критерий?"),
        Check(check_sut_for_L1, 
              error="Для L1 'Суть бизнес-задачи' должна быть 'Проверка доменных знаний'"),
        Check(check_id_version_requirement, 
              error="id версии модели по БМ обязательно для L2"),
        Check(check_domain_nullable_only_for_L0, 
              error="Домен знаний может быть пустым только для уровня знаний L0"),
        Check(check_confidence_interval_for_accuracy,
              error="Для метрики ACCURACY доверительный интервал обязателен и должен быть в формате '(float, float)'"),
        Check(check_marketing_model_name,
              error="Маркетинговое наименование GigaChat должно быть из разрешённого списка"),
        Check(check_model_name,
              error="Наименование модели GigaChat должно быть из разрешённого списка"),
    ],

    strict=False
)


def validate_data(df):
    try:
        validated = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        error_df = err.failure_cases

        print("="*50)
        print("ОШИБКИ В ДАННЫХ".center(50))
        print("="*50)

        grouped_errors = error_df.groupby("check")

        for column, errors in grouped_errors:
            print(f"\nОшибка: {column}")
            for value in errors["failure_case"].unique():
                if errors["column"].any():
                    print(f"Недопустимое значение: {value}")
                    
            print("-"*50)

        print("\n" + "="*50)
        raise ValueError("Обнаружены ошибки в данных") from err

    validated = apply_official_values(validated)
    
    return validated
