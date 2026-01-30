import argparse
import os
import json
import random
import traceback

import numpy as np
import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentation evaluation using JSON dataset with Aerial-R1"
    )
    parser.add_argument(
        "--data-dir",
        default="../data/VSAI_ref/test/images", 
        help="Directory containing the actual image files",
    )
    parser.add_argument(
        "--json-file",
        default="../data/VSAI_ref/test/ref_annotations.json",
        help="Path to the annotation JSON file",
    )
    parser.add_argument(
        "--work-dir",
        default="./results/Aerial_r1_task1",
        help="Directory to save results (masks, visualizations, json)",
    )
    parser.add_argument(
        "--model",
        default="./models/Aerial-r1",
        help="Path to segmentation model (Sa2VA-InternVL3-2B)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=-1,
        help="Number of images to sample, -1 for all images",
    )
    parser.add_argument(
        "--mask-thr",
        type=float,
        default=0.5,
        help="Threshold to binarize predicted masks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    return parser.parse_args()


def parse_custom_json_gt(item_mask):

    objects = []
    
    if not isinstance(item_mask, list):
        return objects

    for obj in item_mask:
        if isinstance(obj, list) and len(obj) > 0:
            raw_coords = obj[0]
            
            if len(raw_coords) == 8:
                pts = np.array(raw_coords, dtype=np.float32).reshape((-1, 2))
                pts = np.round(pts).astype(np.int32)
                objects.append({"points": pts})
                
            elif len(raw_coords) == 4:
                x, y, w, h = raw_coords
                x1, y1 = int(round(x)), int(round(y))
                x2, y2 = int(round(x + w)), int(round(y + h))
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                objects.append({"points": pts})
                
    return objects


def create_gt_masks(gt_objects, img_h, img_w):

    gt_masks = []
    for obj in gt_objects:
        pts = obj["points"]
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        cv2.fillPoly(mask, [pts], 1)
        
        gt_masks.append(mask)
    return gt_masks


def merge_masks(masks_list, img_h, img_w):
    if not masks_list:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    
    merged_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for mask in masks_list:
        binary_mask = (mask > 0).astype(np.uint8)
        merged_mask = np.logical_or(merged_mask, binary_mask).astype(np.uint8)
    
    return merged_mask


def mask_iou(mask1, mask2):
    m1 = mask1 > 0
    m2 = mask2 > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0 if np.sum(m1) == 0 and np.sum(m2) == 0 else 0.0
    return float(inter) / float(union)


def binarize_and_resize_pred_mask(pred_mask, img_h, img_w, thr=0.5):
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


def draw_visualization_box_vs_mask(
    image,
    gt_objects,
    merged_pred_mask,
    overall_iou,
    prompt_text,
    save_path,
    alpha=0.4,
):

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()

    pred_mask_bool = merged_pred_mask.astype(bool)
    overlay[pred_mask_bool] = (0, 0, 255) 
    
    blended = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

    for obj in gt_objects:
        pts = obj["points"]
        cv2.polylines(blended, [pts], True, (0, 255, 0), 2)

    cv2.putText(blended, f"IoU: {overall_iou:.4f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(blended, "Green: GT Box (Rotated) | Red: Pred Mask", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    display_text = prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text
    cv2.putText(blended, f"Prompt: {display_text}", (10, img_bgr.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, blended)

def calculate_pixel_metrics(pred_mask, gt_mask):

    pred = pred_mask > 0
    gt = gt_mask > 0

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, np.logical_not(gt)).sum()
    FN = np.logical_and(np.logical_not(pred), gt).sum()
    TN = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    pixel_acc = (TP + TN) / (TP + FP + FN + TN + 1e-6)

    precision = TP / (TP + FP + 1e-6)

    recall = TP / (TP + FN + 1e-6)

    f1_score = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "pixel_acc": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score)
    }


def main():
    cfg = parse_args()

    os.makedirs(cfg.work_dir, exist_ok=True)
    vis_dir = os.path.join(cfg.work_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Loading annotations from: {cfg.json_file}")
    with open(cfg.json_file, 'r', encoding='utf-8') as f:
        dataset_items = json.load(f)
    
    if not dataset_items:
        print("Error: JSON file is empty.")
        return

    if cfg.sample > 0:
        random.seed(cfg.seed)
        dataset_items = random.sample(dataset_items, min(cfg.sample, len(dataset_items)))

    print(f"Processing {len(dataset_items)} images...")

    print(f"Loading model from: {cfg.model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype="auto", device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    all_overall_ious = []
    all_metrics = {
        "pixel_acc": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }
    all_results = []

    for idx, item in enumerate(dataset_items):
        image_filename = item.get("image")
        text_list = item.get("text", [])
        mask_data = item.get("mask", [])
        
        image_path = os.path.join(cfg.data_dir, image_filename)
        
        print(f"\n[{idx + 1}/{len(dataset_items)}] Processing: {image_filename}")

        if not os.path.exists(image_path):
            print(f"  Error: Image not found at {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue
        
        img_w, img_h = image.size

        gt_objects = parse_custom_json_gt(mask_data)
        
        gt_masks = create_gt_masks(gt_objects, img_h, img_w)

        description = text_list[0] if text_list else "object"
        prompt = f"<image>Please segment the specific vehicle: {description}"
        print(f"  Prompt: {prompt}")

        try:
            result = model.predict_forward(
                image=image,
                text=prompt,
                tokenizer=tokenizer,
            )
        except Exception as e:
            print(f"  Error during inference: {e}")
            traceback.print_exc()
            continue

        pred_masks_raw = result.get("prediction_masks", [])
        prediction_text = result.get("prediction", "")

        pred_masks_bin = []
        if pred_masks_raw is not None and len(pred_masks_raw) > 0:
            for pm in pred_masks_raw:
                binary = binarize_and_resize_pred_mask(pm, img_h, img_w, thr=cfg.mask_thr)
                pred_masks_bin.append(binary)
        
        merged_gt_mask = merge_masks(gt_masks, img_h, img_w)
        merged_pred_mask = merge_masks(pred_masks_bin, img_h, img_w)
        iou = mask_iou(merged_gt_mask, merged_pred_mask)
        metrics = calculate_pixel_metrics(merged_pred_mask, merged_gt_mask)
        
        print(f"  GT: {len(gt_objects)}, Pred: {len(pred_masks_bin)}")
        print(f"  [Metrics] IoU: {iou:.4f} | F1: {metrics['f1_score']:.4f} | "
              f"Prec: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
              f"Acc: {metrics['pixel_acc']:.4f}")

        for k, v in metrics.items():
            all_metrics[k].append(v)

        vis_name = os.path.splitext(image_filename)[0] + "_vis.png"
        vis_path = os.path.join(vis_dir, vis_name)
        draw_visualization_box_vs_mask(
            image=image,
            gt_objects=gt_objects,
            merged_pred_mask=merged_pred_mask,
            overall_iou=iou,
            prompt_text=description,
            save_path=vis_path,
        )

        all_results.append({
            "image": image_filename,
            "prompt": description,
            "prediction_text": prediction_text,
            "iou": iou,
            "num_gt": len(gt_objects),
            "num_pred": len(pred_masks_bin)
        })
        all_overall_ious.append(iou)

    if all_overall_ious:
        mean_iou = float(np.mean(all_overall_ious))
        
        mean_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}

        print("\n" + "="*40)
        print(f"Evaluation Finished.")
        print(f"Mean IoU       : {mean_iou:.4f}")
        print(f"Mean F1-Score  : {mean_metrics['f1_score']:.4f}")
        print(f"Mean Precision : {mean_metrics['precision']:.4f}")
        print(f"Mean Recall    : {mean_metrics['recall']:.4f}")
        print(f"Mean Pixel Acc : {mean_metrics['pixel_acc']:.4f}")
        print("="*40)
    
    json_result_path = os.path.join(cfg.work_dir, "final_metrics_full.json")
    with open(json_result_path, "w", encoding="utf-8") as f:
        json.dump({
            "mean_iou": mean_iou if all_overall_ious else 0,
            "mean_metrics": mean_metrics if all_overall_ious else {},
            "details": all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {json_result_path}")


if __name__ == "__main__":
    main()