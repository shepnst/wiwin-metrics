from typing import List

import evaluate
import numpy as np
import pandas as pd

import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer  # Стеммер для русского языка
from nltk.corpus import stopwords
nlp = spacy.load('en_core_web_lg')
nltk.download('stopwords')
stemmer = SnowballStemmer("russian")
# стоп-слова для русского языка
russian_stop_words = set(stopwords.words('russian'))

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
bertscore = evaluate.load("bertscore")

# Categories
campuses = ["Москва", "Нижний Новгород", "Санкт-Петербург", "Пермь"]
education_levels = ["бакалавриат", "магистратура", "специалитет", "аспирантура"]
question_categories = [
"Деньги",
"Учебный процесс",
"Практическая подготовка",
"ГИА",
"Траектории обучения",
"Английский язык",
"Цифровые компетенции",
"Перемещения студентов / Изменения статусов студентов",
"Онлайн-обучение",
"Цифровые системы",
"Обратная связь",
"Дополнительное образование",
"Безопасность",
"Наука",
"Социальные вопросы",
"ВУЦ",
"Общежития",
"ОВЗ",
"Внеучебка",
"Выпускникам",
"Другое"
]

def get_indexes(campus: str, education_level: str, question_category: List[str])->List[List[int]]:
    campus_index = campuses.index(campus)
    education_level_index = education_levels.index(education_level)
    question_category_indexes = []
    for i in question_category:
        question_category_indexes.append(question_categories.index(i))
    print(campus, education_level, question_category, campus_index, education_level_index, question_category_indexes)
    return [[campus_index, education_level_index, i] for i in question_category_indexes]

def context_recall(ground_truth: str, contexts: List[str])->float:
    """
    Calc rouge btw contexts and ground truth.
    Interpretation: ngram match (recall) btw contexts and desired answer.

    ROUGE - https://huggingface.co/spaces/evaluate-metric/rouge

    return: average rouge for all contexts.
    """
    rs = []
    for c in contexts:
        rs.append(
            rouge.compute(
                predictions=[str(c)],
                references=[str(ground_truth)],
            )["rouge2"]
        )

    return np.mean(rs)

def context_precision(ground_truth: str, contexts: List[str])->float:
    """
    Calc blue btw contexts and ground truth.
    Interpretation: ngram match (precision) btw contexts and desired answer.

    BLEU - https://aclanthology.org/P02-1040.pdf
    max_order - max n-grams to count

    return: average bleu (precision2, w/o brevity penalty) for all contexts.
    """
    bs = []
    for c in contexts:

        try:
            bs.append(
                bleu.compute(
                    predictions=[str(c)],
                    references=[str(ground_truth)],
                    max_order=2,
                )["precisions"][1]
            )
        except ZeroDivisionError:
            bs.append(0)

    return np.mean(bs)

def answer_correctness_literal(
    ground_truth: str,
    answer: str,
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 1,
)->float:
    """
    Calc chrF btw answer and ground truth.
    Interpretation: lingustic match btw answer and desired answer.

    chrF - https://aclanthology.org/W15-3049.pdf
    char_order - n-gram length for chars, default is 6 (from the article)
    word_order - n-gram length for words (chrF++), default is 2 (as it outperforms simple chrF)
    beta - recall weight, beta=1 - simple F1-score

    return: chrF for answ and gt.
    """

    score = chrf.compute(
        predictions=[str(answer)],
        references=[str(ground_truth)],
        word_order=word_order,
        char_order=char_order,
        beta=beta,
    )["score"]

    return score

def answer_correctness_neural(
    ground_truth: str,
    answer: str,
    model_type: str = "cointegrated/rut5-base",
)->float:
    """
    Calc bertscore btw answer and ground truth.
    Interpretation: semantic cimilarity btw answer and desired answer.

    BertScore - https://arxiv.org/pdf/1904.09675.pdf
    model_type - embeds model  (default t5 as the best from my own research and experience)

    return: bertscore-f1 for answ and gt.
    """

    score = bertscore.compute(
        predictions=[str(answer)],
        references=[str(ground_truth)],
        batch_size=1,
        model_type=model_type,
        num_layers=11,
    )["f1"]

    return score

def answer_satisfaction(
    satisfaction: str
)->int:
    if satisfaction == "yes": return 1
    return 0

class Validator:
    """
    Расчет простых метрик качества для заданного датасета.
    """

    questions = {}
    scores = {'general_score': 0.0,
              'context_recall': 0.0,
              'context_precision': 0.0,
              'answer_correctness_literal': 0.0,
              'answer_correctness_neural': 0.0,
              'answer_satisfaction': 0.0}
    particular_scores = [[[0 for i in question_categories] for j in education_levels] for k in campuses]
    particular_number_of_data = [[[0 for h in question_categories] for m in education_levels] for n in campuses]
    number_of_data = 0

    def __init__(
        self,
        neural: bool = False,
    ):
        """
        param neural: есть гпу или нет. По дефолту ее нет(
        """
        self.neural = neural

    def score_sample(
        self,
        answer: str,
        ground_truth: str,
        context: List[str],
        satisfaction: str
    ):
        """
        Расчет для конкретного сэмпла в тестовом датасете.
        """
        scores = {}
        scores["context_recall"] = context_recall(
                ground_truth,
                context,
            ) * 100
        scores["context_precision"] = context_precision(
                ground_truth,
                context,
            ) * 100
        scores["answer_correctness_literal"] = answer_correctness_literal(
                ground_truth=ground_truth,
                answer=answer,
            )
        if self.neural:
            scores["answer_correctness_neural"] = answer_correctness_neural(
                    ground_truth=ground_truth,
                    answer=answer,
                ) * 100
        else:
            scores["answer_correctness_neural"] = 0.0
        scores["answer_satisfaction"] = answer_satisfaction(
                satisfaction
            ) * 100
        return scores

    def frequency_of_question(self, question: str):

        question_ = question.lower()
        tokens = word_tokenize(question_, language='russian')
        tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in russian_stop_words]
        question_ = ' '.join(tokens)
        question_compare = nlp(question_)
        count = 0

        for key, value in Validator.questions.items():
            if key.similarity(question_compare) > 0.7:
                count += 1
                value += 1
        if count == 0:
            Validator.questions[question] = 1

        if len(Validator.questions) >= 50:
            for key, value in Validator.questions.items():
                if value == 1:
                    del Validator.questions[key]

    def validate_rag(
        self,
        new_data,
    ):
        """
        param test_set: пандас датасет с нужными полями: answer, ground_truth, context, question
        """

        gt = new_data['ground_truth']
        answer = new_data['answer']
        context = new_data['contexts']
        satisfaction = new_data['satisfaction']
        question = new_data['question']
        res = self.score_sample(answer, gt, context, satisfaction)
        for k, v in res.items():
            Validator.scores[k] = (Validator.scores[k]*Validator.number_of_data + v) / (Validator.number_of_data + 1)
            for i, j, d in get_indexes(new_data['campus'],
                                       new_data['education_level'],
                                       new_data['question_categories']):
                new_value = (Validator.particular_scores[i][j][d]*Validator.particular_number_of_data[i][j][d] + v) \
                            / (Validator.particular_number_of_data[i][j][d] + 1)
                Validator.particular_scores[i][j][d] = new_value
                Validator.particular_number_of_data[i][j][d] += 1
        Validator.scores['general_score'] = 0.2 * Validator.scores['context_recall'] + \
                                            0.2 * Validator.scores['context_precision'] + \
                                            0.2 * Validator.scores['answer_correctness_literal'] + \
                                            0.3 * Validator.scores['answer_correctness_neural'] + \
                                            0.1 * Validator.scores['answer_satisfaction']
        Validator.number_of_data += 1
        self.frequency_of_question(question)

        return Validator.scores