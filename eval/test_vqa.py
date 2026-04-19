import argparse
import os
import json
import traceback
import re
import string

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser("Aerial-R1 VQA Gap Eval")
    parser.add_argument("--data-dir", default="../../../data/VSAI/test/images")
    parser.add_argument("--json-file", default="../../../data/VSAI/test/vqa2.json")
    parser.add_argument("--model", default="./models/Aerial-r1-b")
    parser.add_argument("--work-dir", default="./results/b_vqa")
    return parser.parse_args()

def judge(question, gt, pred):
    if pred is None:
        return 0.0

    def normalize(s):
        s = s.lower()
        s = re.sub(r"<\|.*?\|>", "", s)
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = s.strip()
        return s

    gt_n = normalize(gt)
    pred_n = normalize(pred)

    if pred_n == gt_n:
        return 1.0


    return 0.0

def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Loading model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            dtype="auto", 
            device_map="auto", 
            trust_remote_code=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(e)
        return

    scores = []
    results = []

    for idx, item in enumerate(data):
        image_path = os.path.join(args.data_dir, item["image"])
        question = item["question"]
        gt = item["answer"]

        print(f"\n[{idx + 1}/{len(data)}]")
        print("Q:", question)
        print("GT:", gt)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            print("Image load failed.")
            continue

        prompt = (
            "<image>"
            "Please answer the following question about the image. "
            "Do NOT perform segmentation. "
            "Only give a short direct answer.\n"
            f"Question: {question}"
        )

        try:
            with torch.no_grad():
                result = model.predict_forward(
                    image=image,
                    text=prompt,
                    tokenizer=tokenizer,
                )
        except Exception:
            print("Model inference failed:")
            traceback.print_exc()
            continue

        pred = result.get("prediction", "")
        print("Pred:", pred)

        score = judge(question, gt, pred)
        print("Score:", score)

        scores.append(score)
        results.append({
            "image": item["image"],
            "question": question,
            "gt": gt,
            "pred": pred,
            "score": score,
        })

    mean_score = float(np.mean(scores)) if scores else 0.0
    print("\nMean Score:", mean_score)

    with open(os.path.join(args.work_dir, "vqa_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_score": mean_score,
                "details": results
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

if __name__ == "__main__":
    main()