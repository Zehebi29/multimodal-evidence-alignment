"""
LLM Client — 封装硅基流动 Qwen API 调用

设计原则：
- 只负责 API 调用，不负责 prompt 构造或输出解析
- 支持重试、超时、错误处理
- 支持流式/非流式
- 可切换不同 LLM（Qwen/DeepSeek/GPT）
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str                    # 原始输出文本
    model: str                      # 使用的模型
    usage: Dict[str, int] = field(default_factory=dict)  # token 用量
    latency_ms: float = 0.0         # 延迟（毫秒）
    raw: Dict = field(default_factory=dict)  # 原始 API 响应


class LLMClient:
    """LLM API 客户端"""
    
    # 支持的模型配置
    MODELS = {
        "qwen": {
            "provider": "siliconflow",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "api_key_env": "SILICONFLOW_API_KEY",
        },
        "qwen-lite": {
            "provider": "siliconflow",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "api_key_env": "SILICONFLOW_API_KEY",
        },
        "deepseek": {
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "deepseek-flash": {
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-v4-flash",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
    }
    
    def __init__(self, 
                 model: str = "qwen",
                 api_key: str = None,
                 base_url: str = None,
                 timeout: float = 120.0,
                 max_retries: int = 3):
        """
        Args:
            model: 模型名称（qwen/qwen-lite/deepseek）或自定义模型名
            api_key: API Key（优先级高于环境变量）
            base_url: API Base URL（覆盖默认值）
            timeout: 请求超时（秒）
            max_retries: 最大重试次数
        """
        # 解析模型配置
        if model in self.MODELS:
            config = self.MODELS[model]
            self.model_name = config["model"]
            self.base_url = base_url or config["base_url"]
            self.api_key = api_key or self._get_env_key(config["api_key_env"])
        else:
            # 自定义模型
            self.model_name = model
            self.base_url = base_url or "https://api.siliconflow.cn/v1"
            self.api_key = api_key or self._get_env_key("SILICONFLOW_API_KEY")
        
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
    
    @classmethod
    def from_config(cls, config_path: str = None, **overrides):
        """Create LLMClient from a YAML configuration file.
        
        API keys are NEVER read from the config file — they must be set
        via environment variables (SILICONFLOW_API_KEY, DEEPSEEK_API_KEY, etc.).
        
        Args:
            config_path: Path to llm_config.yaml. If None, searches:
                         1. configs/llm_config.yaml (repo root)
                         2. ./llm_config.yaml (CWD)
            **overrides: Override any config field (model, temperature, etc.)
        
        Returns:
            LLMClient
        
        Example:
            llm = LLMClient.from_config()
            llm = LLMClient.from_config("configs/llm_config.yaml", temperature=0.3)
        """
        import os
        from pathlib import Path
        
        # Resolve config path
        if config_path is None:
            # Search: repo configs/ first, then CWD
            repo_root = Path(__file__).parent.parent
            candidates = [
                repo_root / "configs" / "llm_config.yaml",
                Path.cwd() / "llm_config.yaml",
                Path.cwd() / "configs" / "llm_config.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = str(candidate)
                    break
            else:
                raise FileNotFoundError(
                    "Cannot find llm_config.yaml. "
                    "Create one at configs/llm_config.yaml or pass an explicit path."
                )
        
        # Load YAML
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required for config files: pip install pyyaml")
        
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # Resolve provider
        provider = overrides.get("provider", cfg.get("provider", "siliconflow"))
        provider_cfg = cfg.get("providers", {}).get(provider, {})
        
        # Resolve model
        model = overrides.get("model", cfg.get("model"))
        
        # Resolve base_url: config → provider default → None
        base_url = (
            overrides.get("base_url")
            or cfg.get("base_url")
            or provider_cfg.get("base_url")
        )
        
        # Resolve API key env name
        api_key_env = (
            overrides.get("api_key_env")
            or cfg.get("api_key_env")
            or provider_cfg.get("api_key_env")
        )
        
        # Get API key from environment variable
        api_key = None
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(
                    f"Environment variable {api_key_env} is not set.\n"
                    f"Set it via: export {api_key_env}=sk-xxx\n"
                    f"Or pass api_key= directly to the constructor."
                )
        
        # Resolve generation params
        timeout = overrides.get("timeout", cfg.get("timeout", 120))
        max_retries = overrides.get("max_retries", cfg.get("max_retries", 3))
        
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
    
    @staticmethod
    def _get_env_key(env_name: str) -> str:
        """从环境变量获取 API Key"""
        import os
        key = os.environ.get(env_name, "")
        if not key:
            raise ValueError(f"环境变量 {env_name} 未设置，请传入 api_key 参数")
        return key
    
    def _get_client(self):
        """延迟初始化 HTTP 客户端"""
        if self._client is None:
            if httpx is None:
                raise ImportError("请安装 httpx: pip install httpx")
            self._client = httpx.Client(timeout=self.timeout)
        return self._client
    
    def chat(self, 
             messages: List[Dict[str, str]],
             temperature: float = 0.1,
             max_tokens: int = 2048,
             response_format: Dict = None) -> LLMResponse:
        """
        发送对话请求
        
        Args:
            messages: 消息列表 [{"role": "system", "content": "..."}, ...]
            temperature: 温度（临床场景建议 0.1，低随机性）
            max_tokens: 最大输出 token 数
            response_format: 响应格式（如 {"type": "json_object"}）
        
        Returns:
            LLMResponse
        """
        client = self._get_client()
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            start = time.time()
            try:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                latency = (time.time() - start) * 1000
                
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    
                    return LLMResponse(
                        content=content,
                        model=self.model_name,
                        usage=usage,
                        latency_ms=latency,
                        raw=data,
                    )
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning(f"LLM 调用失败 (attempt {attempt+1}): {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM 调用异常 (attempt {attempt+1}): {last_error}")
            
            # 指数退避
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        
        raise RuntimeError(f"LLM 调用失败（{self.max_retries} 次重试后）: {last_error}")
    
    def chat_json(self,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.1,
                  max_tokens: int = 2048) -> Dict[str, Any]:
        """
        对话并解析 JSON 输出
        
        Returns:
            解析后的 dict
        """
        # 直接调用 + 手动提取 JSON（跳过 json_object 格式，DeepSeek 支持不稳定）
        resp = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return self._extract_json(resp.content)
    
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """从文本中提取 JSON"""
        import re
        
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 ```json ... ``` 块
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试提取第一个 { ... } (贪婪)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            snippet = match.group(0)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
            # 尝试修复截断的 JSON：逐层关闭括号
            fixed = snippet
            open_braces = fixed.count('{') - fixed.count('}')
            open_brackets = fixed.count('[') - fixed.count(']')
            # 如果最后一个有效字符是逗号或冒号，去掉
            fixed = re.sub(r'[,:]\s*$', '', fixed.rstrip())
            fixed += ']' * max(0, open_brackets)
            fixed += '}' * max(0, open_braces)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
            # 最后尝试：截断到最后一个完整 key-value 对
            # 找最后一个 "xxx": value 模式并截断到它结束
            kv_match = list(re.finditer(r'"[^"]+"\s*:\s*(?:"[^"]*"|[\d.]+|true|false|null)', fixed))
            if len(kv_match) >= 3:  # 至少有 3 个字段才值得修复
                last_good = kv_match[-1]
                truncated = fixed[:last_good.end()] + '}' * max(0, open_braces)
                # 加上关闭的数组
                open_brackets_after = truncated.count('[') - truncated.count(']')
                truncated += ']' * max(0, open_brackets_after)
                try:
                    return json.loads(truncated)
                except json.JSONDecodeError:
                    pass
        
        raise ValueError(f"无法从 LLM 输出中提取 JSON:\n{text[:500]}")
    
    def close(self):
        """关闭 HTTP 客户端"""
        if self._client:
            self._client.close()
            self._client = None
    
    def __del__(self):
        self.close()
