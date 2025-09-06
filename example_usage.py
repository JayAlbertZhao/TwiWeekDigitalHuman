"""
InputModule 使用示例

这个文件展示了InputModule的基本用法，包括：
1. 基本初始化和使用
2. 配置文件加载
3. 与记忆管理模块集成
"""

from input_module import InputModule


def basic_example():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建InputModule实例
    input_module = InputModule()
    
    # 构建基本prompt
    user_input = "你好，请介绍一下你自己"
    prompt = input_module.build_prompt(user_input)
    
    print(f"用户输入: {user_input}")
    print(f"构建的prompt:\n{prompt}")
    print("-" * 50)


def config_file_example():
    """配置文件使用示例"""
    print("=== 配置文件使用示例 ===")
    
    # 使用配置文件
    config_path = "example_config.json"
    try:
        input_module = InputModule(config_path)
        
        user_input = "什么是机器学习？"
        prompt = input_module.build_prompt(user_input)
        
        print(f"用户输入: {user_input}")
        print(f"使用配置文件构建的prompt:\n{prompt}")
        
    except Exception as e:
        print(f"配置文件加载失败: {e}")
    
    print("-" * 50)


def custom_config_example():
    """自定义配置示例"""
    print("=== 自定义配置示例 ===")
    
    input_module = InputModule()
    
    # 创建自定义配置
    custom_config = {
        "context_template": [
            {
                "module_name": "greeting",
                "segments": [
                    {"type": "text", "value": "问候语：你好！"},
                    {"type": "text", "value": "我是AI助手，很高兴为你服务。"}
                ],
                "importance_func": True
            },
            {
                "module_name": "user_question",
                "segments": [
                    {"type": "text", "value": "用户问题：\n"},
                    {"type": "variable", "value": "current_user_input"}
                ],
                "importance_func": True
            },
            {
                "module_name": "instruction",
                "segments": [
                    {"type": "text", "value": "\n请用简洁明了的语言回答上述问题。"}
                ],
                "importance_func": True
            }
        ]
    }
    
    # 更新配置
    input_module.update_config(custom_config)
    
    # 使用新配置
    user_input = "解释一下什么是深度学习"
    prompt = input_module.build_prompt(user_input)
    
    print(f"用户输入: {user_input}")
    print(f"自定义配置构建的prompt:\n{prompt}")
    print("-" * 50)


def memory_integration_example():
    """记忆集成示例"""
    print("=== 记忆集成示例 ===")
    
    # 创建模拟的记忆管理模块
    class MockMemoryModule:
        def retrieve_short_term_memory(self):
            return [
                "用户: 你好",
                "AI: 你好！我是AI助手。",
                "用户: 今天天气怎么样？",
                "AI: 抱歉，我无法获取实时天气信息。"
            ]
        
        def retrieve_long_term_memory(self, query, top_k=3):
            return [
                ("深度学习是机器学习的一个子领域", 0.92),
                ("神经网络是深度学习的基础", 0.88),
                ("机器学习是人工智能的核心技术", 0.85)
            ]
    
    # 创建InputModule并设置记忆模块
    input_module = InputModule()
    input_module.set_memory_management_module(MockMemoryModule())
    
    # 测试包含记忆的prompt构建
    user_input = "请总结我们之前的对话，并解释一下深度学习"
    prompt = input_module.build_prompt(user_input)
    
    print(f"用户输入: {user_input}")
    print(f"包含记忆的prompt:\n{prompt}")
    print("-" * 50)


def main():
    """主函数"""
    print("InputModule 使用示例\n")
    
    # 运行各种示例
    basic_example()
    config_file_example()
    custom_config_example()
    memory_integration_example()
    
    print("所有示例运行完成！")


if __name__ == "__main__":
    main()
