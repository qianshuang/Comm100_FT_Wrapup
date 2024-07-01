# -*- coding: utf-8 -*

from utils import *

answers = []
json_list = load_json_file("data/test.json")
for index, js_ in enumerate(json_list):
    if index <= 174:
        answers.append(json.loads(js_["Answer"]))

pred_answers = []
for line in read_txt_lines("data/test_res.txt"):
    try:
        pred_answers.append(json.loads(line))
    except:
        # print(line)
        pass
print("answers length: {}, pred_answers length: {}".format(len(answers), len(pred_answers)))

score = 0
for i in range(len(pred_answers)):
    answer = answers[i]
    pred_answer = pred_answers[i]
    print("answer_R: {}\nanswer_P: {}\n".format(answer, pred_answer))

    for key in answer:
        if key in pred_answer and answer[key] == pred_answer[key]:
            score += 1
print(score)
print(score / (3 * len(pred_answers)))
