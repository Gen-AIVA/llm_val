import json
from typing import Dict


def read_json_config(config_path: str) -> Dict[str, str]:
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config_data = json.loads(config_file.read())
        return config_data