import argparse
import os
import json
import random
import traceback
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from pycocoevalcap.rouge.rouge import Rouge

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../../data/VSAI/test/images")
    parser.add_argument("--json-file", default="../../../data/VSAI/test/cap_annotations.json")
    parser.add_argument("--work-dir", default="./results/sft_caption")
    parser.add_argument("--model", default="./models/sft")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    with open(cfg.json_file, 'r', encoding='utf-8') as f:
        dataset_items = json.load(f)
    
    if not dataset_items:
        print("Dataset is empty.")
        return

    if cfg.sample > 0:
        random.seed(cfg.seed)
        dataset_items = random.sample(dataset_items, min(cfg.sample, len(dataset_items)))

    try:
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype="auto", device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    all_results = []
    
    
    gts = {} # ground truths
    res = {} # results / predictions

    for idx, item in enumerate(dataset_items):
        item_id = item.get("id")
        
        
        original_image_path = item.get("image")
        image_filename = os.path.basename(original_image_path) 
        image_path = os.path.join(cfg.data_dir, image_filename)
        
        ground_truth_text = ""
        for conv in item.get("conversations", []):
            if conv.get("from") == "gpt":
                ground_truth_text = conv.get("value")
                break
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            print(f"Error loading image: {image_path}")
            continue
        
        prompt = "<image>Please describe this image in detail."
        for conv in item.get("conversations", []):
            if conv.get("from") == "human":
                prompt = conv.get("value")
                break

        try:
            result = model.predict_forward(
                image=image,
                text=prompt,
                tokenizer=tokenizer,
            )
        except Exception:
            traceback.print_exc()
            continue

        if isinstance(result, dict):
            raw_text = result.get("prediction", "")
        else:
            raw_text = str(result)

        clean_text = raw_text.replace("Sure, ", "").replace("<|im_end|>", "")\
                             .replace("[SEG]", "").replace("</p>", "").replace("<p>", "").strip()

        print(f"[{idx + 1}/{len(dataset_items)}] {image_filename}")
        print(f"  GT  : {ground_truth_text}")
        print(f"  Pred: {clean_text}\n")

        all_results.append({
            "id": item_id,
            "image": image_filename,
            "prompt": prompt,
            "ground_truth": ground_truth_text,
            "prediction_text": clean_text
        })
        
        gts[item_id] = [ground_truth_text]
        res[item_id] = [clean_text]

    print(f"Evaluation Finished. Processed {len(all_results)} images.")
    
    print("\nCalculating ROUGE score...")
    rouge_scorer = Rouge()
    
    avg_rouge_score, scores_list = rouge_scorer.compute_score(gts, res)
    print(f"Average ROUGE-L Score: {avg_rouge_score:.4f}")
    
    
    for i, score in enumerate(scores_list):
        all_results[i]["ROUGE-L_score"] = score
    
    json_result_path = os.path.join(cfg.work_dir, "image_descriptions.json")
    with open(json_result_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "ROUGE-L_average": avg_rouge_score
            },
            "details": all_results
        }, f, indent=2, ensure_ascii=False)
        
    print(f"Results saved to {json_result_path}")

if __name__ == "__main__":
    main()