"""
InputModule 测试文件

该文件演示了InputModule的各种功能，包括：
1. 基本初始化和配置加载
2. 配置文件验证
3. Prompt构建
4. 配置更新
5. 错误处理
"""

import json
import os
from ..input_module import InputModule


def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 创建InputModule实例（使用默认配置）
    input_module = InputModule()
    
    # 验证配置
    is_valid, errors = input_module.validate_config()
    print(f"配置验证结果: {'通过' if is_valid else '失败'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # 测试prompt构建
    test_input = "你好，请介绍一下你自己"
    prompt = input_module.build_prompt(test_input)
    print(f"\n构建的prompt:\n{prompt}")
    
    return input_module


def test_config_file_loading():
    """测试配置文件加载"""
    print("\n=== 测试配置文件加载 ===")
    
    # 使用示例配置文件
    config_path = "example_config.json"
    if os.path.exists(config_path):
        input_module = InputModule(config_path)
        
        # 验证配置
        is_valid, errors = input_module.validate_config()
        print(f"配置文件验证结果: {'通过' if is_valid else '失败'}")
        if errors:
            for error in errors:
                print(f"  - {error}")
        
        # 测试prompt构建
        test_input = "请告诉我今天天气怎么样？"
        prompt = input_module.build_prompt(test_input)
        print(f"\n使用配置文件构建的prompt:\n{prompt}")
        
        return input_module
    else:
        print(f"配置文件 {config_path} 不存在，跳过此测试")
        return None


def test_config_update():
    """测试配置更新"""
    print("\n=== 测试配置更新 ===")
    
    input_module = InputModule()
    
    # 创建新配置
    new_config = {
        "context_template": [
            {
                "module_name": "custom_prompt",
                "segments": [
                    {"type": "text", "value": "这是一个自定义的提示：\n"},
                    {"type": "variable", "value": "current_user_input"},
                    {"type": "text", "value": "\n请回答上述问题。"}
                ],
                "importance_func": True
            }
        ]
    }
    
    # 更新配置
    if input_module.update_config(new_config):
        print("配置更新成功")
        
        # 验证新配置
        is_valid, errors = input_module.validate_config()
        print(f"新配置验证结果: {'通过' if is_valid else '失败'}")
        
        # 测试新配置的prompt构建
        test_input = "什么是人工智能？"
        prompt = input_module.build_prompt(test_input)
        print(f"\n新配置构建的prompt:\n{prompt}")
    else:
        print("配置更新失败")


def test_custom_importance_function():
    """测试自定义重要性函数"""
    print("\n=== 测试自定义重要性函数 ===")
    
    input_module = InputModule()
    
    # 添加自定义重要性函数
    def custom_importance_check(user_input: str) -> bool:
        """自定义重要性检查函数"""
        return len(user_input) > 10  # 只有输入长度大于10才包含
    
    # 动态添加方法
    input_module.custom_importance_check = custom_importance_check
    
    # 创建使用自定义函数的配置
    custom_config = {
        "context_template": [
            {
                "module_name": "conditional_module",
                "segments": [
                    {"type": "text", "value": "这是一个条件性模块，只在输入较长时显示。"}
                ],
                "importance_func": "custom_importance_check"
            },
            {
                "module_name": "always_show",
                "segments": [
                    {"type": "text", "value": "这个模块总是显示。"}
                ],
                "importance_func": True
            }
        ]
    }
    
    input_module.update_config(custom_config)
    
    # 测试短输入
    short_input = "你好"
    prompt_short = input_module.build_prompt(short_input)
    print(f"\n短输入 '{short_input}' 的prompt:\n{prompt_short}")
    
    # 测试长输入
    long_input = "这是一个很长的用户输入，用来测试自定义重要性函数的功能"
    prompt_long = input_module.build_prompt(long_input)
    print(f"\n长输入 '{long_input}' 的prompt:\n{prompt_long}")


def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试无效配置文件路径
    input_module = InputModule("nonexistent_config.json")
    
    # 测试无效配置
    invalid_config = {
        "context_template": [
            {
                "module_name": "invalid_module",
                # 缺少segments字段
                "importance_func": True
            }
        ]
    }
    
    input_module.update_config(invalid_config)
    is_valid, errors = input_module.validate_config()
    print(f"无效配置验证结果: {'通过' if is_valid else '失败'}")
    if errors:
        for error in errors:
            print(f"  - {error}")


def test_memory_integration():
    """测试记忆集成（模拟）"""
    print("\n=== 测试记忆集成（模拟）===")
    
    # 创建模拟的记忆管理模块
    class MockMemoryModule:
        def retrieve_short_term_memory(self):
            return [
                "用户: 你好",
                "AI: 你好！很高兴见到你。",
                "用户: 今天天气怎么样？",
                "AI: 抱歉，我无法获取实时天气信息。"
            ]
        
        def retrieve_long_term_memory(self, query, top_k=3):
            return [
                ("人工智能是计算机科学的一个分支", 0.95),
                ("机器学习是AI的重要技术", 0.87),
                ("深度学习基于神经网络", 0.82)
            ]
    
    # 创建模拟的基础模型接口
    class MockBaseModelInterface:
        def send_prompt(self, prompt):
            print(f"模拟发送prompt到基础模型（长度: {len(prompt)} 字符）")
    
    # 设置模拟模块
    input_module = InputModule()
    input_module.set_memory_management_module(MockMemoryModule())
    input_module.set_base_model_interface(MockBaseModelInterface())
    
    # 测试完整流程
    test_input = "请总结一下我们之前的对话"
    prompt = input_module.process_user_input(test_input)
    print(f"\n完整流程构建的prompt:\n{prompt}")


def main():
    """主测试函数"""
    print("开始测试InputModule...\n")
    
    # 运行各种测试
    test_basic_functionality()
    test_config_file_loading()
    test_config_update()
    test_custom_importance_function()
    test_error_handling()
    test_memory_integration()
    
    print("\n所有测试完成！")


if __name__ == "__main__":
    main()
