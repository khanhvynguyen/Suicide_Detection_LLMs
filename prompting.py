from transformers import pipeline
import os
from typing import List
import torch
from tqdm import tqdm
import pandas as pd
import os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from datetime import datetime
import torch
from collections import Counter
from typing import Optional


def run_prompting(df: pd.DataFrame, checkpoint_path: str) -> pd.DataFrame:
    def datetime_now() -> str:
        # time_format = default(time_format, "%Y-%b-%d %H:%M:%S.%f")
        time_format = "%Y-%b-%d__%H-%M-%S"
        return datetime.now().strftime(time_format)

    path = "logs"
    os.makedirs(path, exist_ok=True)
    log = open(f"{path}/promting_{datetime_now()}.txt", "w")

    def print_and_log(print_string, log=log):
        print("{}".format(print_string))
        log.write("{}\n".format(print_string))
        log.flush()

    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

        ## confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        ## classification report
        cr = classification_report(y_true, y_pred)
        return acc, f1, cm, cr

    def extract_levels(res: str) -> Optional[List[str]]:
        """
        result will be like "Yes, Yes, No"
        """
        try:
            level1, level2, level3 = res.split(",")
            ## strip to remove white space
            level1 = level1.strip()
            level2 = level2.strip()
            level3 = level3.strip()
            levels = [level1, level2, level3]
        except:
            print_and_log("=====  Can't split the result!!!")
            print_and_log(f"res={res}")
            return None
        return levels

    def extract_res_3(res: str) -> int:
        """
        result will be like "Yes, Yes, No"
        """
        levels = extract_levels(res)
        if levels is None:
            return -1

        level1, level2, level3 = levels
        if level3 == "yes":
            return 3
        if level2 == "yes":
            return 2
        if level1 == "yes":
            return 1
        return 0

    def get_answer_one_question(ans: str, parse_type: str, levels: List[str] = None) -> int:
        """
        ans: str, the answer from the model
        parse_type: one of {"initial_answer", "move_on"}
        """
        if parse_type == "move_on":
            ans = ans.lower().split("answer:")[-1].split("-")[0].strip()[:5]
            if "yes" in ans:
                if levels[1] == "yes":
                    return 2
                elif levels[0] == "yes":
                    return 1
                else:
                    return 0
            return 3

        elif parse_type == "initial_answer":
            ans = ans.lower().replace("*", "")
            pattern = "collect answers from the analysis:"

            yes_no_str = ans.split(pattern)[-1].split("-")[0].strip()
            final_ans = extract_res_3(yes_no_str)
            return final_ans
        else:
            raise ValueError("parse_type must be one of {initial_answer, move_on}")

    with open("prompts/prompt_jul15_v3.txt", "r") as file:
        prompt_template = file.read()

    with open("prompts/move_on.txt", "r") as file:
        prompt_template_move_on = file.read()

    pipe = pipeline(
        "text-generation",
        model=checkpoint_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    if pipe.tokenizer.eos_token_id is None:  # fix qwen tokenizer
        pipe.tokenizer.eos_token = "<|endoftext|>"
    if pipe.tokenizer.pad_token_id is None and pipe.tokenizer.eos_token_id is not None:  # add pad token
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    def run_post(post: str, true_label: int, prompt_template: str, parse_type: str, levels: List[str] = None):
        prompt = prompt_template.replace("INPUT_POST_PLACEHOLDER", post)
        print_and_log(f"post: {post}")
        print_and_log(f"num tokens: {len(pipe.tokenizer(prompt)['input_ids'])}\n-------")

        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            pipe.tokenizer.convert_tokens_to_ids("Input"),
            pipe.tokenizer.convert_tokens_to_ids("-----"),
        ]
        terminators = [t for t in terminators if t is not None]

        outputs = pipe(
            [prompt],
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=False,
        )
        answer = outputs[0][0]["generated_text"][len(prompt) :]

        print_and_log(f"answer: {answer}")
        res = get_answer_one_question(answer, parse_type=parse_type, levels=levels)
        print_and_log(f"predicted label={res}")
        print_and_log(f"true label={true_label}")
        print_and_log("-" * 20 + "\n")
        return answer, res

    results = []
    res_dict = {"post": [], "answer": [], "true_label": [], "predicted_label": [], "is_correct": []}
    y_true_list = []

    for index in tqdm(range(len(df))):
        post = df.iloc[index]["post"]
        if "label" in df.columns:
            true_label = df.iloc[index]["label"]
        else:
            true_label = -1

        answer, res = run_post(post, true_label, prompt_template=prompt_template, parse_type="initial_answer")

        if res == 3:
            ans = answer.lower().replace("*", "")
            pattern = "collect answers from the analysis:"

            yes_no_str = ans.split(pattern)[-1].split("-")[0].strip()
            levels = extract_levels(yes_no_str)
            new_answer, new_res = run_post(
                post, true_label, prompt_template=prompt_template_move_on, parse_type="move_on", levels=levels
            )
            if new_res != res:
                print_and_log(f"flipped!!! \ninitial res: {res}")
                print_and_log(f"move_on - new res: {new_res}")
                res = new_res

        res_dict["post"].append(post)
        res_dict["answer"].append(answer)
        res_dict["true_label"].append(true_label)
        res_dict["predicted_label"].append(res)
        if true_label == res:
            res_dict["is_correct"].append("Correct")
            print_and_log("Correct")
        else:
            res_dict["is_correct"].append("Incorrect")
            print_and_log("Incorrect")

        results.append(res)
        y_true_list.append(true_label)

        acc, f1, cm, cr = get_metrics(y_true_list, results)

        print_and_log(f"accuracy: {acc}")
        print_and_log(f"f1 score: {f1}")
        print_and_log(f"confusion matrix: \n {cm}")
        print_and_log(f"classification report: \n {cr}")

    if len(res_dict["is_correct"]) == 0:
        del res_dict["is_correct"]
    final_df = pd.DataFrame(res_dict)

    y_true = y_true_list
    y_pred = results

    print_and_log(f"y_true: {Counter(y_true)}")
    print_and_log(f"y_pred: {Counter(results)}")

    # # save final_df
    # final_df.to_csv(
    #     f"prompting_model/prompt_outputs/res_df/final_df_100_test_2_{datetime_now()}.csv", index=False
    # )

    pred = final_df["predicted_label"].map(
        {-1: "cannot_parse", 0: "indicator", 1: "ideation", 2: "behavior", 3: "attempt"}
    )
    submit = pd.DataFrame({"index": range(len(pred)), "suicide risk": pred})
    # submit.to_csv(f"qwen2--{datetime_now()}.csv", index=False)
    return submit


if __name__ == "__main__":
    df_test = pd.read_csv("data/raw_data/test_set.csv")
    # df_val = pd.read_csv("data/cv_data/all500_val1to5.csv")

    checkpoint_path = "../Qwen2-72B-Instruct"
    run_prompting(df_test, checkpoint_path)
