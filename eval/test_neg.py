import argparse
import os
import json
import traceback

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models using the filtered hard negative test set.")
    parser.add_argument(
        "--data-dir",
        default="../../../data/VSAI/test/images", 
        help="Directory containing the original image files",
    )
    parser.add_argument(
        "--test-json",
        default="./hard_negative_annotations.json",
        help="Path to the test labeling file (the hard negative benchmark)",
    )
    parser.add_argument(
        "--work-dir",
        default="./results/sft_l_neg",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--model",
        default="./models/sft-l",
        help="Path to the target segmentation model",
    )
    parser.add_argument(
        "--mask-thr",
        type=float,
        default=0.5,
        help="Threshold to binarize predicted masks",
    )
    return parser.parse_args()

def binarize_and_resize_pred_mask(pred_mask, img_h, img_w, thr=0.5):
    """处理模型输出的 Tensor/List Mask 到原图尺寸"""
    if hasattr(pred_mask, "cpu"):
        mask_np = pred_mask.cpu().numpy()
    else:
        mask_np = np.array(pred_mask)

    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze()

    if mask_np.shape != (img_h, img_w):
        mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    binary = (mask_np > thr).astype(np.uint8)
    return binary

def main():
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    # 1. 加载测试基准数据
    print(f"Loading hard negative test set from: {cfg.test_json}")
    if not os.path.exists(cfg.test_json):
        print(f"Error: Test JSON not found at {cfg.test_json}")
        return
        
    with open(cfg.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    print(f"Total test cases to evaluate: {len(test_data)}")

    # 2. 加载目标模型 (与提供的参考代码完全一致)
    print(f"Loading model from: {cfg.model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype="auto", device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 统计指标
    total_count = 0
    pass_count = 0  # 没画出东西（正确抵抗了幻觉）
    fail_count = 0  # 画出了东西（产生了幻觉）
    total_hallucinated_pixels = 0
    detailed_logs = []

    # 3. 逐个进行测试
    print("\nStarting evaluation on Hard Negatives...")
    for item in tqdm(test_data):
        image_filename = item['image']
        target_prompt = item['text']
        
        image_path = os.path.join(cfg.data_dir, image_filename)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
            
        img_w, img_h = image.size

        # =========================================================
        # [推理核心] 使用与参考代码相同的接口
        # =========================================================
        try:
            with torch.no_grad():
                result = model.predict_forward(
                    image=image,
                    text=target_prompt,
                    tokenizer=tokenizer,
                )
        except Exception as e:
            print(f"Error inference on {image_filename} with prompt '{target_prompt}': {e}")
            traceback.print_exc()
            continue

        # 获取预测的 mask 列表
        pred_masks_raw = result.get("prediction_masks", [])
        
        # 合并所有预测 Mask
        merged_pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        if pred_masks_raw is not None and len(pred_masks_raw) > 0:
            for pm in pred_masks_raw:
                binary = binarize_and_resize_pred_mask(pm, img_h, img_w, thr=cfg.mask_thr)
                merged_pred_mask = np.logical_or(merged_pred_mask, binary).astype(np.uint8)

        # --- 判定逻辑 ---
        pred_pixels = np.sum(merged_pred_mask)
        total_count += 1
        
        if pred_pixels == 0:
            # 成功：在难例面前依然没有画图
            pass_count += 1
            status = "PASS"
        else:
            # 失败：被难例骗出幻觉了
            fail_count += 1
            total_hallucinated_pixels += pred_pixels
            status = "FAIL"

        detailed_logs.append({
            "image": image_filename,
            "prompt": target_prompt,
            "status": status,
            "hallucinated_pixels": int(pred_pixels)
        })

    # 4. 汇总与保存结果
    if total_count > 0:
        pass_rate = (pass_count / total_count) * 100
        avg_pixels = (total_hallucinated_pixels / fail_count) if fail_count > 0 else 0
        
        print("\n" + "="*50)
        print("Hard Negative Benchmark Evaluation Finished")
        print("="*50)
        print(f"Total Test Cases      : {total_count}")
        print(f"Passed (Clean Mask)   : {pass_count}")
        print(f"Failed (Hallucinated) : {fail_count}")
        print("-" * 30)
        print(f"Anti-Hallucination Rate (Pass Rate): {pass_rate:.2f}%")
        print(f"Avg Hallucination Size (Pixels)    : {avg_pixels:.0f} px")
        print("="*50)

        results_data = {
            "summary": {
                "total": total_count,
                "pass": pass_count,
                "fail": fail_count,
                "pass_rate": pass_rate,
                "avg_fail_pixels": avg_pixels
            },
            "details": detailed_logs
        }
        
        save_path = os.path.join(cfg.work_dir, "benchmark_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
            
        print(f"\nDetailed benchmark results saved to: {save_path}")

if __name__ == "__main__":
    main()