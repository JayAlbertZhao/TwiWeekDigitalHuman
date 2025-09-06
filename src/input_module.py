"""
输入模块 (Input Module)

该模块负责接收用户的原始文本输入，并通过其内部的上下文管理系统，
整合长短期记忆，构建发送给基础模型（LLM）的完整prompt。

主要功能：
1. 加载和解析JSON配置文件
2. 管理模块化上下文模板
3. 集成记忆管理模块API
4. 构建发送给基础模型的完整prompt
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputModule:
    """
    输入模块类，负责管理用户输入和上下文，构建发送给基础模型的prompt
    
    主要职责：
    - 加载和解析配置文件
    - 管理上下文模板
    - 集成记忆管理
    - 构建完整prompt
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化InputModule
        
        Args:
            config_path: JSON配置文件的路径，如果为None则使用默认配置
        """
        self.config = None
        self.context_template = []
        self.memory_management_module = None
        self.base_model_interface = None
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.load_default_config()
            logger.info("使用默认配置初始化InputModule")
    
    def load_config(self, config_path: str) -> bool:
        """
        加载JSON配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            bool: 配置加载是否成功
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 解析上下文模板
            if 'context_template' in self.config:
                self.context_template = self.config['context_template']
                logger.info(f"成功加载配置文件: {config_path}")
                logger.info(f"加载了 {len(self.context_template)} 个上下文模板模块")
                return True
            else:
                logger.error("配置文件中缺少 'context_template' 字段")
                return False
                
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {config_path}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON格式错误: {e}")
            return False
        except Exception as e:
            logger.error(f"加载配置文件时发生错误: {e}")
            return False
    
    def load_default_config(self):
        """加载默认配置"""
        self.config = {
            "context_template": [
                {
                    "module_name": "system_prompt",
                    "segments": [
                        {"type": "text", "value": "你是一个友好的AI助手，请根据上下文提供有用的回答。"}
                    ],
                    "importance_func": True
                },
                {
                    "module_name": "short_term_memory",
                    "segments": [
                        {"type": "text", "value": "最近的对话历史：\n"},
                        {"type": "variable", "value": "short_term_history_content"}
                    ],
                    "importance_func": True
                },
                {
                    "module_name": "long_term_memory",
                    "segments": [
                        {"type": "text", "value": "相关的长期记忆：\n"},
                        {"type": "variable", "value": "long_term_memory_content"}
                    ],
                    "importance_func": "is_relevant_long_term_memory"
                },
                {
                    "module_name": "user_input",
                    "segments": [
                        {"type": "text", "value": "用户输入：\n"},
                        {"type": "variable", "value": "current_user_input"}
                    ],
                    "importance_func": True
                }
            ]
        }
        self.context_template = self.config['context_template']
        logger.info("已加载默认配置")
    
    def set_memory_management_module(self, memory_module):
        """
        设置记忆管理模块
        
        Args:
            memory_module: 记忆管理模块实例
        """
        self.memory_management_module = memory_module
        logger.info("记忆管理模块已设置")
    
    def set_base_model_interface(self, base_model_interface):
        """
        设置基础模型接口
        
        Args:
            base_model_interface: 基础模型接口实例
        """
        self.base_model_interface = base_model_interface
        logger.info("基础模型接口已设置")
    
    def is_relevant_long_term_memory(self, user_input: str) -> bool:
        """
        判断是否需要包含长期记忆
        
        Args:
            user_input: 用户输入
            
        Returns:
            bool: 是否需要包含长期记忆
        """
        # 简单的启发式规则：如果用户输入包含特定关键词，则包含长期记忆
        relevant_keywords = ['之前', '以前', '记得', '历史', '总结', '回顾']
        return any(keyword in user_input for keyword in relevant_keywords)
    
    def get_variable_value(self, variable_name: str, user_input: str = "") -> str:
        """
        根据变量名获取对应的值
        
        Args:
            variable_name: 变量名
            user_input: 用户输入
            
        Returns:
            str: 变量的值
        """
        if variable_name == "current_user_input":
            return user_input
        
        elif variable_name == "short_term_history_content":
            if self.memory_management_module:
                try:
                    short_term_memory = self.memory_management_module.retrieve_short_term_memory()
                    if short_term_memory:
                        return "\n".join(short_term_memory[-5:])  # 最近5条记录
                    else:
                        return "无"
                except Exception as e:
                    logger.error(f"获取短期记忆失败: {e}")
                    return "获取失败"
            else:
                return "记忆管理模块未设置"
        
        elif variable_name == "long_term_memory_content":
            if self.memory_management_module and user_input:
                try:
                    long_term_results = self.memory_management_module.retrieve_long_term_memory(
                        user_input, top_k=3
                    )
                    if long_term_results:
                        content = []
                        for text, score in long_term_results:
                            content.append(f"- {text} (相关度: {score:.2f})")
                        return "\n".join(content)
                    else:
                        return "无相关长期记忆"
                except Exception as e:
                    logger.error(f"获取长期记忆失败: {e}")
                    return "获取失败"
            else:
                return "无相关长期记忆"
        
        else:
            # 尝试调用对应的方法
            if hasattr(self, variable_name) and callable(getattr(self, variable_name)):
                try:
                    method = getattr(self, variable_name)
                    if variable_name.startswith('is_'):
                        return method(user_input)
                    else:
                        return method()
                except Exception as e:
                    logger.error(f"调用方法 {variable_name} 失败: {e}")
                    return f"方法调用失败: {e}"
            else:
                logger.warning(f"未知变量: {variable_name}")
                return f"未知变量: {variable_name}"
    
    def should_include_module(self, module: Dict, user_input: str = "") -> bool:
        """
        判断是否应该包含某个模块
        
        Args:
            module: 模块配置
            user_input: 用户输入
            
        Returns:
            bool: 是否应该包含该模块
        """
        importance_func = module.get("importance_func", True)
        
        if isinstance(importance_func, bool):
            return importance_func
        
        elif isinstance(importance_func, str):
            # 调用指定的方法
            if hasattr(self, importance_func) and callable(getattr(self, importance_func)):
                try:
                    method = getattr(self, importance_func)
                    return method(user_input)
                except Exception as e:
                    logger.error(f"调用重要性函数 {importance_func} 失败: {e}")
                    return False
            else:
                logger.warning(f"重要性函数 {importance_func} 不存在")
                return False
        
        else:
            logger.warning(f"无效的重要性函数类型: {type(importance_func)}")
            return False
    
    def build_prompt(self, user_input: str) -> str:
        """
        构建发送给基础模型的完整prompt
        
        Args:
            user_input: 用户输入
            
        Returns:
            str: 构建好的完整prompt
        """
        prompt_parts = []
        
        for module in self.context_template:
            if not self.should_include_module(module, user_input):
                logger.debug(f"跳过模块: {module.get('module_name', 'unknown')}")
                continue
            
            module_content = []
            for segment in module.get("segments", []):
                segment_type = segment.get("type")
                segment_value = segment.get("value")
                
                if segment_type == "text":
                    module_content.append(segment_value)
                elif segment_type == "variable":
                    variable_value = self.get_variable_value(segment_value, user_input)
                    module_content.append(str(variable_value))
                else:
                    logger.warning(f"未知的段类型: {segment_type}")
            
            if module_content:
                module_text = "".join(module_content)
                prompt_parts.append(module_text)
        
        final_prompt = "\n\n".join(prompt_parts)
        logger.info(f"构建的prompt长度: {len(final_prompt)} 字符")
        logger.debug(f"构建的prompt: {final_prompt}")
        
        return final_prompt
    
    def process_user_input(self, user_input: str) -> str:
        """
        处理用户输入，构建prompt并发送给基础模型
        
        Args:
            user_input: 用户输入
            
        Returns:
            str: 构建好的prompt
        """
        if not user_input.strip():
            logger.warning("收到空的用户输入")
            return ""
        
        # 构建prompt
        prompt = self.build_prompt(user_input)
        
        # 如果设置了基础模型接口，则发送prompt
        if self.base_model_interface:
            try:
                # 这里假设基础模型接口有send_prompt方法
                if hasattr(self.base_model_interface, 'send_prompt'):
                    self.base_model_interface.send_prompt(prompt)
                    logger.info("prompt已发送给基础模型接口")
                else:
                    logger.warning("基础模型接口没有send_prompt方法")
            except Exception as e:
                logger.error(f"发送prompt给基础模型接口失败: {e}")
        
        return prompt
    
    def update_config(self, new_config: Dict) -> bool:
        """
        更新配置
        
        Args:
            new_config: 新的配置字典
            
        Returns:
            bool: 更新是否成功
        """
        try:
            if 'context_template' in new_config:
                self.config = new_config
                self.context_template = new_config['context_template']
                logger.info("配置已更新")
                return True
            else:
                logger.error("新配置中缺少 'context_template' 字段")
                return False
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False
    
    def get_config(self) -> Dict:
        """
        获取当前配置
        
        Returns:
            Dict: 当前配置
        """
        return self.config.copy()
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        验证当前配置的有效性
        
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        if not self.config:
            errors.append("配置为空")
            return False, errors
        
        if 'context_template' not in self.config:
            errors.append("缺少 'context_template' 字段")
            return False, errors
        
        context_template = self.config['context_template']
        if not isinstance(context_template, list):
            errors.append("'context_template' 必须是列表")
            return False, errors
        
        for i, module in enumerate(context_template):
            if not isinstance(module, dict):
                errors.append(f"模块 {i} 必须是字典")
                continue
            
            if 'module_name' not in module:
                errors.append(f"模块 {i} 缺少 'module_name' 字段")
            
            if 'segments' not in module:
                errors.append(f"模块 {i} 缺少 'segments' 字段")
            else:
                segments = module['segments']
                if not isinstance(segments, list):
                    errors.append(f"模块 {i} 的 'segments' 必须是列表")
                else:
                    for j, segment in enumerate(segments):
                        if not isinstance(segment, dict):
                            errors.append(f"模块 {i} 的段 {j} 必须是字典")
                        elif 'type' not in segment:
                            errors.append(f"模块 {i} 的段 {j} 缺少 'type' 字段")
                        elif 'value' not in segment:
                            errors.append(f"模块 {i} 的段 {j} 缺少 'value' 字段")
                        elif segment['type'] not in ['text', 'variable']:
                            errors.append(f"模块 {i} 的段 {j} 的 'type' 必须是 'text' 或 'variable'")
        
        return len(errors) == 0, errors


# 示例用法和测试代码
if __name__ == "__main__":
    # 创建InputModule实例
    input_module = InputModule()
    
    # 验证配置
    is_valid, errors = input_module.validate_config()
    if is_valid:
        print("配置验证通过")
    else:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
    
    # 测试prompt构建
    test_input = "你好，请介绍一下你自己"
    prompt = input_module.build_prompt(test_input)
    print(f"\n构建的prompt:\n{prompt}")
    
    # 测试配置更新
    new_config = {
        "context_template": [
            {
                "module_name": "simple_prompt",
                "segments": [
                    {"type": "text", "value": "用户说："},
                    {"type": "variable", "value": "current_user_input"}
                ],
                "importance_func": True
            }
        ]
    }
    
    if input_module.update_config(new_config):
        print("\n配置更新成功")
        new_prompt = input_module.build_prompt(test_input)
        print(f"新配置构建的prompt:\n{new_prompt}")
