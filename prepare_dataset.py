# 1_prepare_data.py
import os, shutil, xml.etree.ElementTree as ET
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image

# === ตั้งค่า ===
RAW_IMG = 'archive/images'
RAW_XML = 'archive/annotations'
YOLO_ROOT = 'yolov8_dataset'
os.makedirs(f'{YOLO_ROOT}/images/train', exist_ok=True)
os.makedirs(f'{YOLO_ROOT}/images/val', exist_ok=True)
os.makedirs(f'{YOLO_ROOT}/images/test', exist_ok=True)
os.makedirs(f'{YOLO_ROOT}/labels/train', exist_ok=True)
os.makedirs(f'{YOLO_ROOT}/labels/val', exist_ok=True)
os.makedirs(f'{YOLO_ROOT}/labels/test', exist_ok=True)

classes = ['without_mask', 'with_mask']
class_map = {name: i for i, name in enumerate(classes)}

def xml_to_yolo(xml_path, w, h):
    tree = ET.parse(xml_path)
    lines = []
    for obj in tree.findall('object'):
        name = obj.find('name').text
        if name not in class_map: continue
        cls = class_map[name]
        box = obj.find('bndbox')
        xmin, ymin = float(box.find('xmin').text), float(box.find('ymin').text)
        xmax, ymax = float(box.find('xmax').text), float(box.find('ymax').text)
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

# === ดึงไฟล์ ===
img_files = glob(f"{RAW_IMG}/*.png") + glob(f"{RAW_IMG}/*.jpg")
xml_files = glob(f"{RAW_XML}/*.xml")
img_names = {os.path.splitext(os.path.basename(f))[0] for f in img_files}
xml_names = {os.path.splitext(os.path.basename(f))[0] for f in xml_files}
common = sorted(img_names & xml_names)
print(f"พบไฟล์ตรงกัน: {len(common)}")

# === แบ่ง train/val/test ===
train, temp = train_test_split(common, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)
splits = {'train': train, 'val': val, 'test': test}

# === แปลงและคัดลอก ===
for split, names in splits.items():
    for name in names:
        # คัดลอกภาพ
        src_img = f"{RAW_IMG}/{name}.png"
        if not os.path.exists(src_img):
            src_img = src_img.replace('.png', '.jpg')
        dst_img = f"{YOLO_ROOT}/images/{split}/{name}.jpg"
        shutil.copy(src_img, dst_img)

        # แปลง XML → TXT
        xml_path = f"{RAW_XML}/{name}.xml"
        with Image.open(src_img) as img:
            w, h = img.size
        lines = xml_to_yolo(xml_path, w, h)
        with open(f"{YOLO_ROOT}/labels/{split}/{name}.txt", 'w') as f:
            f.write('\n'.join(lines))

# === สร้าง data.yaml ===
yaml = f"""
train: ./images/train
val: ./images/val
test: ./images/test
nc: 2
names: ['without_mask', 'with_mask']
"""
with open(f"{YOLO_ROOT}/data.yaml", 'w', encoding='utf-8') as f:
    f.write(yaml)

print("เตรียมข้อมูลเสร็จ! พร้อมฝึกโมเดล")