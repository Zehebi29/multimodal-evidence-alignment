from __future__ import annotations
"""
多模态 RAG — 视觉+临床特征的 embedding 检索

流程:
1. 裁剪 YOLO bbox → 病灶区域图片
2. 视觉 embedding: Qwen3-VL-Embedding-8B (病灶图片)
3. 文本 embedding: Qwen3-Embedding-0.6B (临床特征文本)
4. 多模态融合: 加权拼接
5. 余弦相似度检索 Top-K → few-shot 注入

支持两种向量后端:
- backend="json" (默认): 内存列表 + JSON 序列化，零额外依赖
- backend="faiss" (可选): FAISS IVF 索引，需 pip install faiss-cpu
"""

import os
import json
import base64
import time
import numpy as np
from io import BytesIO
from typing import Optional

try:
    from PIL import Image
except ImportError:
    Image = None

import requests

from harness.slots.backends import (
    CaseEmbedding, IndexBackend, create_backend
)


class EvidenceStore:
    """多模态 RAG 证据存储"""

    def __init__(self, api_key: str,
                 vision_model: str = "Qwen/Qwen3-VL-Embedding-8B",
                 text_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 base_url: str = "https://api.siliconflow.cn/v1",
                 vision_weight: float = 0.6,
                 text_weight: float = 0.4,
                 top_k: int = 3,
                 backend: str = "json"):
        self.api_key = api_key
        self.vision_model = vision_model
        self.text_model = text_model
        self.base_url = base_url
        self.vision_weight = vision_weight
        self.text_weight = text_weight
        self.top_k = top_k
        self.backend: IndexBackend = create_backend(backend)
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    # ── 裁剪 ──

    def crop_lesion(self, image_path: str, bbox: list, padding: float = 0.15):
        """从原图裁剪病灶区域，带 padding"""
        if Image is None:
            raise ImportError("Pillow is required: pip install Pillow")
        try:
            img = Image.open(image_path)
        except Exception:
            return None

        w, h = img.size
        x1, y1, x2, y2 = bbox
        # 添加 padding
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * padding), int(bh * padding)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        return img.crop((x1, y1, x2, y2))

    def crop_best_detection(self, case_id: str, detections: dict,
                            dataset_dir: str):
        """从检测结果中取置信度最高的 bbox，裁剪"""
        frames = detections.get("frames", [])
        if not frames:
            return None

        # 找置信度最高的检测
        best_det = None
        best_conf = 0
        best_frame_id = None
        for frame in frames:
            for det in frame.get("detections", []):
                conf = det.get("conf", 0)
                if conf > best_conf and det.get("bbox"):
                    best_conf = conf
                    best_det = det
                    best_frame_id = frame.get("frame_id", "")

        if not best_det:
            return None

        # 找对应图片
        frames_dir = os.path.join(dataset_dir, "cases", case_id, "frames")
        frame_file = best_frame_id if best_frame_id.endswith(".jpg") else f"{best_frame_id}.jpg"
        img_path = os.path.join(frames_dir, frame_file)

        if not os.path.exists(img_path):
            # fallback: 找第一张
            if os.path.isdir(frames_dir):
                imgs = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
                if imgs:
                    img_path = os.path.join(frames_dir, imgs[0])
                else:
                    return None
            else:
                return None

        return self.crop_lesion(img_path, best_det["bbox"])

    # ── Embedding API ──

    def _get_vision_embedding(self, image, text: str = "") -> Optional[np.ndarray]:
        """视觉 embedding: 图片(+可选文本) → 向量（带 retry）"""
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = text if text else "EUS lesion image"

        for attempt in range(3):
            try:
                resp = self._session.post(
                    f"{self.base_url}/embeddings",
                    json={
                        "model": self.vision_model,
                        "input": prompt,
                        "images": [f"data:image/jpeg;base64,{img_b64}"],
                    },
                    timeout=90,
                )
                data = resp.json()
                if "data" in data:
                    return np.array(data["data"][0]["embedding"], dtype=np.float32)
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"  [WARN] Vision embedding failed after 3 attempts: {e}")
        return None

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """文本 embedding → 向量（带 retry）"""
        for attempt in range(3):
            try:
                resp = self._session.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.text_model, "input": text},
                    timeout=60,
                )
                data = resp.json()
                if "data" in data:
                    return np.array(data["data"][0]["embedding"], dtype=np.float32)
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"  [WARN] Text embedding failed after 3 attempts: {e}")
        return None

    # ── 特征文本化 ──

    def _features_to_text(self, preds: list, features: dict) -> str:
        """将 AI 预测+临床特征转为文本"""
        from collections import Counter
        classes = Counter(p["class"] for p in preds)
        pred_str = ", ".join(f"{cls}({cnt})" for cls, cnt in classes.most_common())
        avg_conf = sum(p["conf"] for p in preds) / max(len(preds), 1)

        parts = [f"AI predictions: {pred_str}, avg confidence: {avg_conf:.0%}"]
        if features.get("location"):
            parts.append(f"Location: {features['location']}")
        if features.get("age"):
            parts.append(f"Age: {features['age']}")
        if features.get("gender"):
            parts.append(f"Gender: {features['gender']}")
        return ". ".join(parts)

    # ── 向量融合 ──

    def _fuse_vectors(self, visual: Optional[np.ndarray],
                      text: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """视觉+文本向量加权融合（L2 归一化后加权拼接）

        当某一模态缺失时用零向量填充，保证所有融合向量维度一致。
        """
        if visual is None and text is None:
            return None

        # 确定维度（首次调用时缓存）
        vis_dim = 4096   # Qwen3-VL-Embedding-8B
        txt_dim = 1024   # Qwen3-Embedding-0.6B

        if visual is None:
            v_norm = np.zeros(vis_dim, dtype=np.float32)
        else:
            v_norm = self._normalize(visual)

        if text is None:
            t_norm = np.zeros(txt_dim, dtype=np.float32)
        else:
            t_norm = self._normalize(text)

        fused = np.concatenate([
            v_norm * self.vision_weight,
            t_norm * self.text_weight,
        ])
        return fused

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    # ── 索引构建 ──

    def add_to_index(self, case_id: str, image,
                     preds: list, features: dict, ground_truth: str,
                     evidence_chain: dict = None):
        """将一个病例的多模态 embedding 加入索引（含证据链）"""
        text_desc = self._features_to_text(preds, features)

        # 视觉 embedding
        visual_vec = None
        if image is not None:
            visual_vec = self._get_vision_embedding(image, text_desc)

        # 文本 embedding
        text_vec = self._get_text_embedding(text_desc)

        # 融合
        fused_vec = self._fuse_vectors(visual_vec, text_vec)

        ec = evidence_chain or {}
        entry = CaseEmbedding(
            case_id=case_id,
            ground_truth=ground_truth,
            ai_predictions=preds,
            features=features,
            visual_vector=visual_vec,
            text_vector=text_vec,
            fused_vector=fused_vec,
            reasoning=ec.get("reasoning", ""),
            supporting_evidence=ec.get("supporting_evidence", []),
            opposing_evidence=ec.get("opposing_evidence", []),
            differential=ec.get("differential", []),
            uncertainty_sources=ec.get("uncertainty_sources", []),
        )
        self.backend.add(entry)

    # ── 检索 ──

    def search(self, query_image,
               query_preds: list, query_features: dict,
               k: int = None) -> list[tuple[float, CaseEmbedding]]:
        """检索最相似的 k 个病例"""
        if k is None:
            k = self.top_k

        # 构建查询向量
        text_desc = self._features_to_text(query_preds, query_features)
        q_visual = None
        if query_image is not None:
            q_visual = self._get_vision_embedding(query_image, text_desc)
        q_text = self._get_text_embedding(text_desc)
        q_fused = self._fuse_vectors(q_visual, q_text)

        if q_fused is None:
            return []

        # 委托后端检索
        results = self.backend.search(q_fused, k)
        return [(sim, self.backend.get(idx)) for sim, idx in results]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(dot / (na * nb))

    # ── 构建 few-shot ──

    def build_few_shot(self, results: list[tuple[float, CaseEmbedding]]) -> str:
        """将检索结果构建为 few-shot 文本（含证据链推理）"""
        lines = ["Reference Cases (with evidence chain reasoning):"]
        for i, (sim, case) in enumerate(results):
            from collections import Counter
            classes = Counter(p["class"] for p in case.ai_predictions)
            pred_str = ", ".join(f"{cls}({cnt})" for cls, cnt in classes.most_common())
            avg_conf = sum(p["conf"] for p in case.ai_predictions) / max(len(case.ai_predictions), 1)
            consistency = max(classes.values()) / max(sum(classes.values()), 1) if classes else 0
            loc = case.features.get("location", "?")

            lines.append(f"")
            lines.append(f"  --- Case {i+1}: {case.case_id} (similarity={sim:.2f}) ---")
            lines.append(f"  AI predictions: {pred_str}, avg conf {avg_conf:.0%}, consistency {consistency:.0%}")
            lines.append(f"  Location: {loc}")
            lines.append(f"  Final diagnosis: {case.ground_truth}")

            # 证据链
            if case.reasoning:
                lines.append(f"  Reasoning: {case.reasoning[:300]}")
            if case.supporting_evidence:
                ev_str = "; ".join(str(e)[:100] for e in case.supporting_evidence[:3])
                lines.append(f"  Supporting evidence: {ev_str}")
            if case.opposing_evidence:
                ev_str = "; ".join(str(e)[:100] for e in case.opposing_evidence[:3])
                lines.append(f"  Counter evidence: {ev_str}")
            if case.differential:
                diff_str = "; ".join(
                    f"{d.get('diagnosis','')}({d.get('probability','')})" if isinstance(d, dict) else str(d)[:80]
                    for d in case.differential[:3]
                )
                lines.append(f"  Differential: {diff_str}")
            if case.uncertainty_sources:
                lines.append(f"  Uncertainty: {', '.join(case.uncertainty_sources[:3])}")

        return "\n".join(lines)

    # ── 向量重算 ──

    def recompute_fused_vectors(self):
        """Recompute fused vectors using current weights (local, no API calls)."""
        for i in range(len(self.backend)):
            entry = self.backend.get(i)
            entry.fused_vector = self._fuse_vectors(entry.visual_vector, entry.text_vector)

    # ── 持久化 ──

    def save_index(self, path: str):
        """保存索引到文件（委托后端）"""
        self.backend.save(path)

    def load_index(self, path: str):
        """从文件加载索引（委托后端）"""
        self.backend.load(path)

    @property
    def index(self) -> list:
        """兼容旧代码：返回所有条目列表（只读）"""
        return [self.backend.get(i) for i in range(len(self.backend))]
