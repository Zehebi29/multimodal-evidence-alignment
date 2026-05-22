"""
Harness 配置 — 所有可调参数集中管理

设计原则：
- 所有算法参数从配置读取，不硬编码
- 参数变更有版本记录
- 支持实验对比（不同参数集）
"""

from dataclasses import dataclass, field
from typing import Dict
import yaml
from pathlib import Path


@dataclass
class SufficiencyConfig:
    """充分性计算配置"""
    # 权重
    weight_feature_coverage: float = 0.3
    weight_consistency: float = 0.4
    weight_confidence: float = 0.3
    
    # 阈值
    threshold_high: float = 0.7
    threshold_low: float = 0.4


@dataclass
class FeatureImportanceConfig:
    """特征重要性配置"""
    # 从 Meta Harness 实验得出的经验值
    # 后续应由数据驱动更新
    importance: Dict[str, float] = field(default_factory=lambda: {
        "lesion_location": 0.9,
        "layer_origin": 0.8,
        "echo_pattern": 0.6,
        "homogeneous": 0.4,
    })


@dataclass
class HarnessConfig:
    """Harness 总配置"""
    version: str = "1.0.0"
    description: str = "默认配置"
    
    sufficiency: SufficiencyConfig = field(default_factory=SufficiencyConfig)
    feature_importance: FeatureImportanceConfig = field(default_factory=FeatureImportanceConfig)
    
    # 数据集信息
    dataset_version: str = "smt_v1"
    model_version: str = "v5m_five_frame"
    
    # 随机种子（可复现性）
    random_seed: int = 42
    
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "description": self.description,
            "sufficiency": {
                "weights": {
                    "feature_coverage": self.sufficiency.weight_feature_coverage,
                    "consistency": self.sufficiency.weight_consistency,
                    "confidence": self.sufficiency.weight_confidence,
                },
                "thresholds": {
                    "high": self.sufficiency.threshold_high,
                    "low": self.sufficiency.threshold_low,
                },
            },
            "feature_importance": self.feature_importance.importance,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "random_seed": self.random_seed,
        }
    
    @classmethod
    def from_yaml(cls, path: str) -> "HarnessConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        config.version = data.get("version", config.version)
        config.description = data.get("description", config.description)
        
        if "sufficiency" in data:
            s = data["sufficiency"]
            if "weights" in s:
                config.sufficiency.weight_feature_coverage = s["weights"].get("feature_coverage", 0.3)
                config.sufficiency.weight_consistency = s["weights"].get("consistency", 0.4)
                config.sufficiency.weight_confidence = s["weights"].get("confidence", 0.3)
            if "thresholds" in s:
                config.sufficiency.threshold_high = s["thresholds"].get("high", 0.7)
                config.sufficiency.threshold_low = s["thresholds"].get("low", 0.4)
        
        if "feature_importance" in data:
            config.feature_importance.importance = data["feature_importance"]
        
        config.dataset_version = data.get("dataset_version", config.dataset_version)
        config.model_version = data.get("model_version", config.model_version)
        config.random_seed = data.get("random_seed", config.random_seed)
        
        return config
    
    def save_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


# 默认配置
DEFAULT_CONFIG = HarnessConfig()


# 预定义配置集
CONFIGS = {
    "default": HarnessConfig(
        version="1.0.0",
        description="默认配置 — 基于 Exp01-06 实验",
    ),
    "conservative": HarnessConfig(
        version="1.0.1",
        description="保守配置 — 更高阈值，更少误判",
        sufficiency=SufficiencyConfig(
            threshold_high=0.8,
            threshold_low=0.5,
        ),
    ),
    "aggressive": HarnessConfig(
        version="1.0.2",
        description="激进配置 — 更低阈值，更多诊断",
        sufficiency=SufficiencyConfig(
            threshold_high=0.6,
            threshold_low=0.3,
        ),
    ),
}
