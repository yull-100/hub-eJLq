import os
import shutil
from pdf2image import convert_from_path
import dashscope
from dashscope import MultiModalConversation

# === 步骤 1: PDF 转图片 ===
pdf_path = "test.pdf"
# poppler_dir = os.path.dirname(shutil.which("pdftoppm"))  # 自动获取 poppler 路径

images = convert_from_path(pdf_path, first_page=1, last_page=1, poppler_path="/opt/homebrew/bin")
image_path = "temp_page.jpg"
images[0].save(image_path, "JPEG")

# === 步骤 2: 调用 Qwen-VL OCR ===
dashscope.api_key = "sk-685d5da74c2047dbb7d80c0e80fcb05d"

response = MultiModalConversation.call(
    model="qwen-vl-ocr",
    messages=[{
        'role': 'user',
        'content': [{'image': f"file://{os.path.abspath(image_path)}"}]
    }]
)

if response.status_code == 200:
    text = response.output.choices[0].message.content
    print("识别结果:\n", text)
else:
    print("API 调用失败:", response.message)