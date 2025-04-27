import os
import requests
import zipfile
from tqdm import tqdm

# COCO 2017 데이터셋의 검증 이미지 및 주석 데이터 URL
urls = {
    'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

# 파일을 저장할 디렉토리 설정
save_dir = './coco2017'
os.makedirs(save_dir, exist_ok=True)

def download_with_progress(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted {os.path.basename(zip_path)} to {extract_to}")

# 각 URL에 대해 다운로드 및 압축 해제 수행
for key, url in urls.items():
    file_name = url.split('/')[-1]
    save_path = os.path.join(save_dir, file_name)
    
    print(f"⬇️ Downloading {file_name} ...")
    download_with_progress(url, save_path)
    
    print(f"📦 Extracting {file_name} ...")
    extract_zip(save_path, save_dir)
