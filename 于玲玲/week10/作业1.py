# clip_zh_zero_shot.py
from PIL import Image
import torch
from collections import defaultdict
from transformers import AltCLIPModel, AltCLIPProcessor

print("正在加载中文多模态模型 BAAI/AltCLIP...")
model = AltCLIPModel.from_pretrained("/Users/yull/AI/model/BAAI/AltCLIP")
processor = AltCLIPProcessor.from_pretrained("/Users/yull/AI/model/BAAI/AltCLIP")

# === 中文提示模板===
templates_zh = [
    "一张{}的照片",
    "一张模糊的{}的照片",
    "一张裁剪过的{}的照片",
    "一张清晰的{}的照片",
    "一张{}的特写照片",
    "一张质量差的{}的照片",
    "一幅{}的画",
    "一幅{}的绘画",
    "一张高分辨率的{}的照片",
    "一张低分辨率的{}的照片"
]

# === 定义你的分类类别（中文）===
# 建议包含“其他”以避免强行二选一
classes = ["狗", "猫", "汽车", "其他"]

# 构建所有提示文本
texts = []
label_map = []
for cls in classes:
    for template in templates_zh:
        texts.append(template.format(cls))
        label_map.append(cls)

print(f"共生成 {len(texts)} 个中文提示（每类 {len(templates_zh)} 个）")

# === 加载本地图片 ===
image_path = ("dog.jpg")
try:
    image = Image.open(image_path).convert("RGB")
    print(f"已加载图片: {image_path} | 尺寸: {image.size}")
except FileNotFoundError:
    print(f"错误：未找到图片 '{image_path}'")
    exit(1)

# === 预处理 + 推理 ===
inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # [1, num_texts]
    probs = logits_per_image.softmax(dim=1)      # 归一化概率

# === 聚合每个类别的平均概率 ===
agg_probs = defaultdict(float)
for i, label in enumerate(label_map):
    agg_probs[label] += probs[0][i].item()

# 归一化（确保总和为1）
total = sum(agg_probs.values())
for label in agg_probs:
    agg_probs[label] /= total

# === 输出结果 ===
print("中文零样本分类结果：")
for label, prob in sorted(agg_probs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {label:6} : {prob:.2%}")

# === 决策逻辑 ===
best_label = max(agg_probs, key=agg_probs.get)
confidence = agg_probs[best_label]

if confidence > 0.6:
    print(f"\n最终预测：{best_label}（置信度 {confidence:.2%}）")
else:
    print(f"\n不确定：最高置信度仅 {confidence:.2%}，可能是未知类别")

# === 可选：显示 top-3 提示词（用于调试）===
print("\nTop-3 匹配的提示词：")
top3_indices = probs[0].argsort(descending=True)[:3]
for idx in top3_indices:
    print(f"  '{texts[idx]}' → {probs[0][idx]:.3f} | 类别: {label_map[idx]}")