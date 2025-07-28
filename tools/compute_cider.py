# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 23:35
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : compute_cider.py

import argparse

from cocap.modeling.eval_captioning import evaluate
from cocap.utils.json import load_json


def main(opt):
    pred = load_json(opt.pred_json)
    ref = load_json(opt.ref_json)

    if opt.raw:
        pred_results = pred
    else:
        pred_results = pred["results"]

    if opt.lower:
        pred_results = {
            k: [{kk: vv.lower()[:-2] + "." if vv.endswith(" .") else vv.lower() for kk, vv in x.items()} for x in v]
            for k, v in pred_results.items()
        }
        ref = {k: [vv.lower() for vv in v] for k, v in ref.items()}

    metrics = evaluate(submission={"results": pred_results}, reference=ref)

    print('>>>  Bleu_4: {:.2f} - METEOR: {:.2f} - ROUGE_L: {:.2f} - CIDEr: {:.2f}'.
          format(metrics['Bleu_4'] * 100, metrics['METEOR'] * 100, metrics['ROUGE_L'] * 100,
                 metrics['CIDEr'] * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute CIDEr")
    parser.add_argument("pred_json", type=str)
    parser.add_argument("ref_json", type=str)
    parser.add_argument("--lower", action="store_true")
    parser.add_argument("--raw", action="store_true")

    main(parser.parse_args())
