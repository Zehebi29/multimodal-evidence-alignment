"""
Output Parser — LLM 输出解析与 Schema 验证

Harness 控制 LLM 的第三层：验证 LLM 输出是否符合预期格式。

设计原则：
- 严格 JSON Schema 验证
- 类型转换和默认值填充
- 不合格输出直接拒绝，不让错误传播
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """解析结果"""
    success: bool
    data: Optional[Dict] = None
    errors: List[str] = None
    warnings: List[str] = None


class SchemaValidator:
    """LLM 输出解析器"""
    
    # ── Schema 定义 ──
    
    EVIDENCE_CHAIN_SCHEMA = {
        "required": ["diagnosis", "supporting_evidence", "opposing_evidence"],
        "properties": {
            "diagnosis": {"type": str, "values": ["LM", "GIST", "NET", "EP", "LIP", "uncertain"]},
            "reasoning": {"type": str},
            "supporting_evidence": {"type": list},
            "opposing_evidence": {"type": list},
            "differential": {"type": list},
        },
    }
    
    PATTERN_SCHEMA = {
        "required": ["patterns"],
        "properties": {
            "patterns": {"type": list},
            "summary": {"type": str},
            "priority_ranking": {"type": list},
        },
    }
    
    EVOLUTION_SCHEMA = {
        "required": ["suggestions"],
        "properties": {
            "suggestions": {"type": list},
            "reasoning": {"type": str},
            "config_patches": {"type": dict},
        },
    }
    
    REVIEW_SCHEMA = {
        "required": ["error_type", "root_cause"],
        "properties": {
            "error_type": {"type": str, "values": ["false_positive", "false_negative", "wrong_subtype", "feature_error"]},
            "severity": {"type": str, "values": ["high", "medium", "low"]},
            "root_cause": {"type": str},
            "learning": {"type": str},
            "affected_features": {"type": list},
            "suggested_action": {"type": str},
        },
    }
    
    @classmethod
    def parse_evidence_chain(cls, raw: Dict) -> ParseResult:
        """解析证据链输出"""
        return cls._validate(raw, cls.EVIDENCE_CHAIN_SCHEMA, "evidence_chain")
    
    @classmethod
    def parse_patterns(cls, raw: Dict) -> ParseResult:
        """解析模式发现输出"""
        return cls._validate(raw, cls.PATTERN_SCHEMA, "patterns")
    
    @classmethod
    def parse_evolution(cls, raw: Dict) -> ParseResult:
        """解析进化建议输出"""
        return cls._validate(raw, cls.EVOLUTION_SCHEMA, "evolution")
    
    @classmethod
    def parse_review(cls, raw: Dict) -> ParseResult:
        """解析复核分析输出"""
        return cls._validate(raw, cls.REVIEW_SCHEMA, "review")
    
    @classmethod
    def _validate(cls, data: Any, schema: Dict, schema_name: str) -> ParseResult:
        """通用 Schema 验证"""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            return ParseResult(
                success=False,
                errors=[f"[{schema_name}] 输出不是 dict 类型: {type(data)}"],
            )
        
        # 检查必需字段
        for field in schema.get("required", []):
            if field not in data:
                errors.append(f"[{schema_name}] 缺少必需字段: {field}")
        
        # 检查字段类型和值
        for field, rules in schema.get("properties", {}).items():
            if field not in data:
                continue
            
            value = data[field]
            expected_type = rules.get("type")
            
            # 类型检查
            if expected_type and not isinstance(value, expected_type):
                # 尝试类型转换
                try:
                    if expected_type == (int, float) and isinstance(value, str):
                        value = float(value)
                        data[field] = value
                    elif expected_type == str and value is not None:
                        value = str(value)
                        data[field] = value
                    else:
                        errors.append(f"[{schema_name}] {field} 类型错误: 期望 {expected_type}, 实际 {type(value)}")
                        continue
                except (ValueError, TypeError):
                    errors.append(f"[{schema_name}] {field} 类型转换失败")
                    continue
            
            # 值范围检查
            if "min" in rules and isinstance(value, (int, float)):
                if value < rules["min"]:
                    warnings.append(f"[{schema_name}] {field} 值 {value} 低于最小值 {rules['min']}")
                    data[field] = rules["min"]
            
            if "max" in rules and isinstance(value, (int, float)):
                if value > rules["max"]:
                    warnings.append(f"[{schema_name}] {field} 值 {value} 超过最大值 {rules['max']}")
                    data[field] = rules["max"]
            
            # 枚举值检查
            if "values" in rules:
                if value not in rules["values"]:
                    # 对 diagnosis 做模糊匹配
                    if field == "diagnosis":
                        value = cls._fuzzy_match_diagnosis(str(value), rules["values"])
                        data[field] = value
                        if value is None:
                            errors.append(f"[{schema_name}] {field} 值无法识别: {data.get(field)}")
                    else:
                        warnings.append(f"[{schema_name}] {field} 值不在预期范围: {value}")
        
        # 填充默认值
        defaults = {
            "reasoning": "",
            "supporting_evidence": [],
            "opposing_evidence": [],
            "differential": [],
            "severity": "medium",
        }
        for field, default in defaults.items():
            if field in schema.get("properties", {}) and field not in data:
                data[field] = default
                warnings.append(f"[{schema_name}] 填充默认值: {field}={default}")
        
        success = len(errors) == 0
        if not success:
            logger.warning(f"[{schema_name}] 验证失败: {errors}")
        
        return ParseResult(
            success=success,
            data=data if success else None,
            errors=errors,
            warnings=warnings,
        )
    
    @staticmethod
    def _fuzzy_match_diagnosis(value: str, valid_values: List[str]) -> Optional[str]:
        """模糊匹配诊断类别"""
        value_upper = value.upper().strip()
        
        # 精确匹配
        if value_upper in valid_values:
            return value_upper
        
        # 部分匹配
        mapping = {
            "平滑肌瘤": "LM", "LEIOMYOMA": "LM",
            "胃肠间质瘤": "GIST", "GIST": "GIST",
            "神经内分泌肿瘤": "NET", "NEUROENDOCRINE": "NET",
            "异位胰腺": "EP", "PANCREAS": "EP", "ECTOPIC": "EP",
            "脂肪瘤": "LIP", "LIPOMA": "LIP",
            "不确定": "uncertain", "UNCERTAIN": "uncertain", "UNKNOWN": "uncertain",
        }
        for keyword, code in mapping.items():
            if keyword in value or keyword in value_upper:
                if code in valid_values:
                    return code
        
        return None
