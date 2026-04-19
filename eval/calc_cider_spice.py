import json
import os
from pycocoevalcap.cider.cider import Cider

def main():
    # 1. 读取你已经生成的 JSON 文件 (替换为你实际的文件名)
    json_file_path = "./results/b_caption/image_descriptions.json" 
    
    if not os.path.exists(json_file_path):
        print(f"找不到文件: {json_file_path}")
        return

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gts = {}
    res = {}

    # 2. 格式化为 pycocoevalcap 需要的字典结构: {id: [text]}
    for item in data["details"]:
        item_id = item["id"]
        gts[item_id] = [item["ground_truth"]]
        res[item_id] = [item["prediction_text"]]

    # 3. 仅计算 CIDEr
    print("正在计算 CIDEr...")
    cider_scorer = Cider()
    avg_cider, cider_scores = cider_scorer.compute_score(gts, res)

    # 4. 将分数回填到 JSON 中
    data["metrics"]["CIDEr_average"] = avg_cider

    for i, item in enumerate(data["details"]):
        item["CIDEr_score"] = cider_scores[i]

    # 5. 保存新的 JSON 文件
    output_path = "./results/b_caption/image_descriptions_with_cider.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("-" * 30)
    print("计算完成！")
    print(f"CIDEr Average Score: {avg_cider:.4f}")
    print(f"结果已成功追加并保存至: {output_path}")

if __name__ == "__main__":
    main()