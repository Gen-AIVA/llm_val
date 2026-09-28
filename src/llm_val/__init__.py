import os
from pathlib import Path


cur_path = Path(__file__).resolve().parent
os.environ["chrf"] = str(Path.joinpath(cur_path, "resources/evaluate/chrf"))
os.environ["gigachat_tokenizer"] = str(Path.joinpath(cur_path, "resources/gigachat_tokenizer"))
os.environ["haystack_for_detect"] = str(Path.joinpath(cur_path, "resources/haystack_for_detect"))
os.environ["NLTK_DATA"] = str(Path.joinpath(cur_path, "resources/nltk_data"))
os.environ["ru_spacy"] = str(Path.joinpath(cur_path, "resources/ru_spacy/ru_core_news_lg/ru_core_news_lg-3.8.0"))
os.environ["ru_spacy_sm"] = str(Path.joinpath(cur_path, "resources/ru_spacy/ru_core_news_sm/ru_core_news_sm-3.8.0"))
os.environ["TOLOKA_PATH"] = str(Path.joinpath(cur_path, "resources/toloka/data_toloka.pkl"))