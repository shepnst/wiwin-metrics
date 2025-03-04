import json
import pandas as pd
from func_to_call import parse_all_data, parse_data_with_time
from metrics import ValidatorSimple

data_v1 = parse_all_data('datasets/val_set.json')

data_v2 = parse_data_with_time('datasets/val_set.json')

with open('parsed_tuning.json', 'w', encoding='utf-8') as f:
    json.dump(data_v1, f, ensure_ascii=False)

with open('parsed_dash.json', 'w', encoding='utf-8') as f:
    json.dump(data_v2, f, ensure_ascii=False)

with open('parsed_tuning.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)

formatted_data = []

for item in training_data:
    contexts = "\n".join([ctx['text'] for ctx in item['contexts']])
    base_input = f"Вопрос: {item['user_question']}\nКонтекст: {contexts}"

    if item['winner'] == 'Saiga':
        formatted_data.append({
            "input": base_input,
            "output": item['saiga_answer'],
            "source": "saiga",
            "rating": "good" if item['winner'] in ['Saiga', 'Оба хорошо'] else "bad"
        })

    elif item['winner'] == 'GigaChat':
        formatted_data.append({
            "input": base_input,
            "output": item['giga_answer'],
            "source": "giga",
            "rating": "good" if item['winner'] in ['GigaChat', 'Оба хорошо'] else "bad"
        })

    elif item['winner'] == 'Оба хорошо':
        formatted_data.extend([
            {
                "input": base_input,
                "output": item['saiga_answer'],
                "source": "saiga",
                "rating": "good"
            },
            {
                "input": base_input,
                "output": item['giga_answer'],
                "source": "giga",
                "rating": "good"
            }
        ])

    elif item['winner'] == 'Оба плохо':
        formatted_data.extend([
            {
                "input": base_input,
                "output": item['saiga_answer'],
                "source": "saiga",
                "rating": "bad"
            },
            {
                "input": base_input,
                "output": item['giga_answer'],
                "source": "giga",
                "rating": "bad"
            }
        ])

    else:
        formatted_data.append({
            "input": base_input,
            "output": item['saiga_answer'],
            "source": "unknown",
            "rating": "neutral"
        })


dataset = pd.DataFrame({
    "question": [
        "Какие документы регулируют порядок обслуживания студентов в столовой?",
        "Каковы основные этапы прохождения учебной практики?",
    ],

    "answer": [
        "Я ГЛУПИ МОДЕЛЬ НЕ УМЕЮ...",
        "А Я УМЕЮ...", #ответы вашей модели
    ],
    "ground_truth": [
        "Порядок организации обслуживания регулируется следующими документами...",
        "Основные этапы включают: 1. Размещение программы...", #эталонные ответы
    ],
    "contexts": [
        [
            "сотрудника НИУ ВШЭ с указанием руководителя...",
            "Центр сервиса «Студент» – Национальный исследовательский университет...",
            # остальные контексты для первого вопроса
        ],
        [
            "уточной аттестации и текущего контроля успеваемости студентов...",
            "Траектории обучения в бакалавриате...",
            # ... контексты для второго вопроса
        ],
    ]
})

vs = ValidatorSimple(neural=True)
vs.validate_rag(dataset)

print(formatted_data[0])
