from typing import List

import evaluate
import numpy as np
import pandas as pd

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
bertscore = evaluate.load("bertscore")


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

    scores = {'general_score': 0.0,
              'context_recall': 0.0,
              'context_precision': 0.0,
              'answer_correctness_literal': 0.0,
              'answer_correctness_neural': 0.0,
              'answer_satisfaction': 0.0}
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
        res = self.score_sample(answer, gt, context, satisfaction)
        for k, v in res.items():
            Validator.scores[k] = (Validator.scores[k]*Validator.number_of_data + v) / (Validator.number_of_data + 1)
        Validator.scores['general_score'] = 0.2 * Validator.scores['context_recall'] + \
                                            0.2 * Validator.scores['context_precision'] + \
                                            0.2 * Validator.scores['answer_correctness_literal'] + \
                                            0.3 * Validator.scores['answer_correctness_neural'] + \
                                            0.1 * Validator.scores['answer_satisfaction']
        Validator.number_of_data += 1

        return Validator.scores