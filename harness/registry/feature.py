"""
Feature Registry — 特征注册表管理模块

Multimodal Evidence Alignment (MEA) 的特征获取机制

功能：
- 加载特征注册表配置
- 按 source 类型查询特征
- 验证特征依赖关系
- 处理缺失特征策略

用法：
    from harness.feature_registry import FeatureRegistry
    registry = FeatureRegistry("configs/feature_registry.yaml")
    
    # 获取所有特征
    all_features = registry.get_all_features()
    
    # 按 source 类型查询
    visual_features = registry.get_features_by_source("visual")
    doctor_features = registry.get_features_by_source("doctor")
    
    # 检查特征是否可用
    available = registry.check_availability({"lesion_location": "胃底"})
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class Feature:
    """特征定义"""
    name: str
    source: str
    extractor: Optional[str]
    required: bool
    description: str
    values: List[str]
    default: Any
    depends_on: List[str]
    prompt: Optional[str] = None
    mi_weight: Optional[float] = None  # Mutual Information weight


class FeatureRegistry:
    """特征注册表"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.features: Dict[str, Feature] = {}
        for name, feat_config in self.config.get('features', {}).items():
            self.features[name] = Feature(
                name=name,
                source=feat_config['source'],
                extractor=feat_config.get('extractor'),
                required=feat_config.get('required', False),
                description=feat_config.get('description', ''),
                values=feat_config.get('values', []),
                default=feat_config.get('default'),
                depends_on=feat_config.get('depends_on', []),
                prompt=feat_config.get('prompt'),
                mi_weight=feat_config.get('mi_weight'),
            )
        
        self.source_types = self.config.get('source_types', {})
        self.missing_strategy = self.config.get('missing_strategy', {})
    
    def get_all_features(self) -> Dict[str, Feature]:
        """获取所有特征定义"""
        return self.features.copy()
    
    def get_feature(self, name: str) -> Optional[Feature]:
        """获取指定特征"""
        return self.features.get(name)
    
    def get_features_by_source(self, source: str) -> Dict[str, Feature]:
        """按 source 类型查询特征"""
        return {
            name: feat for name, feat in self.features.items()
            if feat.source == source
        }
    
    def get_required_features(self) -> Dict[str, Feature]:
        """获取所有 required 特征"""
        return {
            name: feat for name, feat in self.features.items()
            if feat.required
        }
    
    def get_optional_features(self) -> Dict[str, Feature]:
        """获取所有 optional 特征"""
        return {
            name: feat for name, feat in self.features.items()
            if not feat.required
        }
    
    def check_availability(self, available_features: Dict[str, Any]) -> Dict[str, str]:
        """
        检查特征可用性，返回缺失特征及其处理策略
        
        Args:
            available_features: 已可用的特征 {name: value}
        
        Returns:
            缺失特征的处理策略 {feature_name: strategy_action}
        """
        missing = {}
        for name, feat in self.features.items():
            if name in available_features:
                continue
            
            if feat.required and feat.default is None:
                missing[name] = 'reject'  # 必须输入，无法默认
            elif feat.required and feat.default is not None:
                missing[name] = 'use_default'  # 使用默认值
            else:
                missing[name] = 'skip'  # 可选，跳过
        
        return missing
    
    def validate_dependencies(self, feature_name: str, available: Dict[str, Any]) -> bool:
        """验证特征依赖是否满足"""
        feat = self.features.get(feature_name)
        if not feat:
            return False
        
        for dep in feat.depends_on:
            if dep not in available:
                return False
        
        return True
    
    def extract(self, feature_name: str, case_dir: str, **kwargs) -> Any:
        """Extract a single feature value from a case directory.
        
        Uses the registered extractor function. Falls back to kwargs['available']
        if extractor is not defined (e.g., doctor-input features).
        
        Args:
            feature_name: Name of the feature to extract
            case_dir: Path to the case directory (contains metadata.json)
            **kwargs: Additional context (e.g., available=dict of pre-loaded values)
        
        Returns:
            Extracted feature value, or None if unavailable
        """
        feat = self.features.get(feature_name)
        if not feat:
            return None
        
        # Try extractor function first
        from harness.registry.extractors import get_extractor
        extractor_fn = get_extractor(feature_name)
        if extractor_fn is not None:
            return extractor_fn(case_dir, **kwargs)
        
        # Fall back to pre-loaded available_features
        available = kwargs.get('available', {})
        if feature_name in available:
            return available[feature_name]
        
        # Use default if available
        if feat.default is not None:
            return feat.default
        
        return None
    
    def extract_all(self, case_dir: str, available: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract all registered features from a case directory.
        
        Args:
            case_dir: Path to the case directory
            available: Pre-loaded feature values (e.g., from doctor input)
        
        Returns:
            Dict of {feature_name: value}
        """
        if available is None:
            available = {}
        result = {}
        for name in self.features:
            value = self.extract(name, case_dir, available=available)
            if value is not None:
                result[name] = value
        return result

    def get_doctor_prompts(self) -> Dict[str, str]:
        """获取需要医生输入的特征及其提示语"""
        return {
            name: feat.prompt
            for name, feat in self.features.items()
            if feat.source == 'doctor' and feat.prompt
        }
    
    def to_summary(self) -> str:
        """生成特征摘要"""
        lines = ["Feature Registry Summary:", ""]
        
        by_source = {}
        for name, feat in self.features.items():
            by_source.setdefault(feat.source, []).append(feat)
        
        for source, feats in by_source.items():
            lines.append(f"  [{source}]")
            for feat in feats:
                req = "required" if feat.required else "optional"
                lines.append(f"    - {feat.name}: {feat.description} ({req})")
            lines.append("")
        
        return "\n".join(lines)


def load_registry(config_path: str = None) -> FeatureRegistry:
    """加载特征注册表的便捷函数"""
    if config_path is None:
        # 默认路径
        config_path = Path(__file__).parent.parent.parent / "configs" / "feature_registry.yaml"
    
    return FeatureRegistry(str(config_path))


if __name__ == "__main__":
    # 测试
    registry = load_registry()
    print(registry.to_summary())
    
    print("\n--- Required Features ---")
    for name, feat in registry.get_required_features().items():
        print(f"  {name}: {feat.description}")
    
    print("\n--- Doctor Input Features ---")
    for name, prompt in registry.get_doctor_prompts().items():
        print(f"  {name}: {prompt}")
    
    print("\n--- Missing Feature Strategy ---")
    test_available = {"lesion_location": "胃底"}  # 只有 location
    missing = registry.check_availability(test_available)
    for feat, action in missing.items():
        print(f"  {feat}: {action}")
