import re
import typing as tp

def parse_html_dialogue(dialogue: str) -> tp.List[str]:
    '''
    Функция для парсинга диалога из формата HTML в список словарей
    Parameters:
        dialogue: HTML строка с диалогом
    Return:
        Список с ролью, фразой и контекстом
    '''
    pattern = r'<span class(?:=["s]?)([^"s>]+)(?:["s]?)>'
    matches = re.findall(pattern, dialogue)
    users = set(matches)
    dialogue = dialogue.replace('<br />', '')
    messages = dialogue.split('</span>')
    result = []
    for msg in messages:
        if '<span class=' in msg:
            txt = msg.split(':')[1].strip()
            txt = txt.replace(',', '')
            txt = re.sub(r'[^\w\s]', '', txt)
            result.append(txt)
    return result