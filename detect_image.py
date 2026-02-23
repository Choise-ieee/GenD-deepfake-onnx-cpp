"""
深度伪造检测推理脚本
"""

import sys
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image

THIS = Path(__file__).resolve()
ROOT = THIS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector import align_face
from src.hf.modeling_gend import GenD as GenD_HF
from src.retinaface import RetinaFace

# ============== 配置 ==============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "yermandy/GenD_CLIP_L_14"
OUTPUT_PATH = "result.jpg"
DETECTOR_URL = "https://huggingface.co/datasets/theanhntp/Liblib/resolve/ae4357741af379482690fe3e0f2fa6fd32ba33b4/insightface/models/buffalo_l/det_10g.onnx"
DETECTOR_PATH = Path("weights/models/buffalo_l/det_10g.onnx")

# 全局缓存
_detector = None
_model = None
_preproc = None


def download_file(url: str, save_path: Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, str(save_path))
    print(f"✅ Saved to: {save_path}")


def load_detector(thresh: float = 0.5):
    global _detector
    if _detector is not None:
        return _detector
    
    if not DETECTOR_PATH.exists():
        download_file(DETECTOR_URL, DETECTOR_PATH)
    
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    _detector = RetinaFace(str(DETECTOR_PATH), providers=providers)
    _detector.prepare(ctx_id=0, nms_thresh=0.4, input_size=(640, 640), det_thresh=thresh)
    print(f"✅ RetinaFace loaded")
    return _detector


def load_model():
    global _model, _preproc
    if _model is not None:
        return _model, _preproc
    
    print(f"Loading model: {MODEL_ID}")
    _model = GenD_HF.from_pretrained(MODEL_ID)
    _model.eval().to(DEVICE)
    _preproc = _model.feature_extractor.preprocess
    print(f"✅ Model loaded on {DEVICE}")
    return _model, _preproc


def infer(image_bgr: np.ndarray, detector, model, preproc, scale: float = 1.3, max_faces: int = None):
    """人脸检测 + 对齐 + 推理"""
    xyxy, landmarks = detector.detect(image_bgr)
    if xyxy is None or len(xyxy) == 0:
        return []
    
    indices = list(range(len(xyxy)))
    indices.sort(key=lambda i: (xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1]), reverse=True)
    if max_faces:
        indices = indices[:max_faces]
    
    results = []
    for i in indices:
        try:
            aligned, _ = align_face(image_bgr, landmarks[i], scale=scale)
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            
            with torch.no_grad():
                tensor = preproc(Image.fromarray(aligned)).unsqueeze(0).to(DEVICE)
                prob = model(tensor).softmax(-1)[0, 1].item()
            
            results.append((xyxy[i], prob))
        except:
            continue
    
    return results


def annotate(image_bgr: np.ndarray, faces):
    """标注结果"""
    vis = image_bgr.copy()
    for bbox, p_fake in faces:
        x1, y1, x2, y2 = map(int, bbox[:4])
        color = (0, int(255 * (1 - p_fake)), int(255 * p_fake))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"fake: {p_fake:.3f}"
        cv2.putText(vis, text, (x1 + 6, max(20, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, text, (x1 + 6, max(20, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def detect(image_path: str, thresh: float = 0.5, scale: float = 1.3, max_faces: int = None):
    """检测单张图片"""
    # 加载
    detector = load_detector(thresh)
    model, preproc = load_model()
    
    # 读取
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    
    # 推理
    faces = infer(img, detector, model, preproc, scale, max_faces)
    
    # 结果
    probs = [p for _, p in faces]
    avg = np.mean(probs) if probs else 0.0
    
    # 标注
    vis = annotate(img, faces)
    
    # 保存
    cv2.imwrite(OUTPUT_PATH, vis)
    
    # 打印
    print(f"\n{'='*40}")
    print(f"Image: {image_path}")
    print(f"Faces: {len(faces)}")
    for i, (_, p) in enumerate(faces):
        print(f"  Face {i+1}: Fake={p:.2%}")
    print(f"Average: Fake={avg:.2%}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"{'='*40}\n")
    
    return {"faces": faces, "avg_fake": avg}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deepfake Detection")
    parser.add_argument("--image", type=str, required=True, help="Image path")
    parser.add_argument("--thresh", type=float, default=0.5, help="Face detection threshold")
    parser.add_argument("--scale", type=float, default=1.3, help="Face alignment scale")
    parser.add_argument("--max-faces", type=int, help="Max faces per image")
    args = parser.parse_args()
    
    detect(args.image, thresh=args.thresh, scale=args.scale, max_faces=args.max_faces)
