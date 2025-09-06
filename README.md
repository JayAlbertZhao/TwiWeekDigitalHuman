# InputModule 输入模块

## 概述

InputModule是一个智能的输入处理模块，负责接收用户的原始文本输入，并通过其内部的上下文管理系统，整合长短期记忆，构建发送给基础模型（LLM）的完整prompt。

## 主要功能

1. **配置文件管理**: 支持JSON格式的配置文件，可动态加载和更新
2. **上下文模板**: 模块化的上下文模板系统，支持条件性包含
3. **记忆集成**: 与记忆管理模块集成，获取短期和长期记忆
4. **智能提示构建**: 根据用户输入和上下文动态构建prompt
5. **配置验证**: 完整的配置验证机制，确保配置文件的正确性

## 安装和依赖

```bash
# 确保Python环境已激活
conda activate DigitalHuman

# 安装依赖
pip install typing-extensions
```

## 快速开始

### 1. 基本使用

```python
from input_module import InputModule

# 创建InputModule实例（使用默认配置）
input_module = InputModule()

# 构建prompt
user_input = "你好，请介绍一下你自己"
prompt = input_module.build_prompt(user_input)
print(prompt)
```

### 2. 使用配置文件

```python
# 使用自定义配置文件
input_module = InputModule("config.json")

# 处理用户输入
prompt = input_module.process_user_input("今天天气怎么样？")
```

### 3. 集成记忆管理模块

```python
# 设置记忆管理模块
input_module.set_memory_management_module(memory_module)

# 设置基础模型接口
input_module.set_base_model_interface(base_model_interface)

# 现在可以自动获取记忆并构建完整prompt
prompt = input_module.process_user_input("请总结我们之前的对话")
```

## 配置文件格式

### 基本结构

```json
{
    "context_template": [
        {
            "module_name": "模块名称",
            "segments": [
                {"type": "text", "value": "固定文本"},
                {"type": "variable", "value": "变量名"}
            ],
            "importance_func": true
        }
    ]
}
```

### 字段说明

- **module_name**: 模块的唯一标识符
- **segments**: 段列表，每个段可以是固定文本或变量
- **importance_func**: 重要性函数，控制模块是否被包含
  - `true`: 总是包含
  - `false`: 总是排除
  - `"方法名"`: 调用指定方法动态判断

### 段类型

- **text**: 固定文本段
- **variable**: 变量段，值由系统动态获取

### 内置变量

- `current_user_input`: 当前用户输入
- `short_term_history_content`: 短期记忆内容
- `long_term_memory_content`: 长期记忆内容

## API 参考

### 主要方法

#### `__init__(config_path=None)`
初始化InputModule实例

**参数:**
- `config_path`: 配置文件路径，可选

#### `load_config(config_path)`
加载JSON配置文件

**参数:**
- `config_path`: 配置文件路径

**返回:**
- `bool`: 加载是否成功

#### `build_prompt(user_input)`
构建发送给基础模型的完整prompt

**参数:**
- `user_input`: 用户输入

**返回:**
- `str`: 构建好的prompt

#### `process_user_input(user_input)`
处理用户输入，构建prompt并发送给基础模型

**参数:**
- `user_input`: 用户输入

**返回:**
- `str`: 构建好的prompt

#### `update_config(new_config)`
更新配置

**参数:**
- `new_config`: 新的配置字典

**返回:**
- `bool`: 更新是否成功

#### `validate_config()`
验证当前配置的有效性

**返回:**
- `Tuple[bool, List[str]]`: (是否有效, 错误信息列表)

### 设置方法

#### `set_memory_management_module(memory_module)`
设置记忆管理模块

#### `set_base_model_interface(base_model_interface)`
设置基础模型接口

## 使用示例

### 示例1: 基本prompt构建

```python
from input_module import InputModule

# 创建实例
input_module = InputModule()

# 构建prompt
prompt = input_module.build_prompt("什么是人工智能？")
print(prompt)
```

### 示例2: 自定义配置

```python
# 自定义配置
custom_config = {
    "context_template": [
        {
            "module_name": "custom_prompt",
            "segments": [
                {"type": "text", "value": "用户问题："},
                {"type": "variable", "value": "current_user_input"},
                {"type": "text", "value": "\n请用中文回答。"}
            ],
            "importance_func": True
        }
    ]
}

# 更新配置
input_module.update_config(custom_config)

# 使用新配置
prompt = input_module.build_prompt("解释量子计算")
```

### 示例3: 条件性模块

```python
# 添加自定义重要性函数
def should_include_weather(user_input):
    weather_keywords = ['天气', '温度', '下雨', '晴天']
    return any(keyword in user_input for keyword in weather_keywords)

input_module.should_include_weather = should_include_weather

# 配置中使用自定义函数
weather_config = {
    "context_template": [
        {
            "module_name": "weather_context",
            "segments": [
                {"type": "text", "value": "天气相关信息：今天天气晴朗，温度25度。"}
            ],
            "importance_func": "should_include_weather"
        }
    ]
}

input_module.update_config(weather_config)
```

## 错误处理

InputModule包含完善的错误处理机制：

- 配置文件不存在或格式错误
- 配置验证失败
- 记忆模块调用失败
- 变量解析错误

所有错误都会记录到日志中，并提供详细的错误信息。

## 测试

运行测试文件来验证功能：

```bash
python test_input_module.py
```

测试包括：
- 基本功能测试
- 配置文件加载测试
- 配置更新测试
- 自定义重要性函数测试
- 错误处理测试
- 记忆集成测试

## 日志

InputModule使用Python的logging模块，可以通过以下方式调整日志级别：

```python
import logging
logging.getLogger('input_module').setLevel(logging.DEBUG)
```

## 扩展性

InputModule设计为高度可扩展的：

1. **自定义重要性函数**: 可以添加任意数量的自定义判断函数
2. **新变量类型**: 可以在`get_variable_value`方法中添加新的变量类型
3. **配置格式**: 支持在配置文件中添加新的字段和设置
4. **模块集成**: 可以轻松集成新的外部模块

## 注意事项

1. 确保配置文件使用UTF-8编码
2. 重要性函数必须返回布尔值
3. 变量名必须与`get_variable_value`方法中的处理逻辑匹配
4. 记忆管理模块和基础模型接口是可选的，但设置后功能更完整

## 贡献

欢迎提交Issue和Pull Request来改进这个模块！

## 许可证

本项目采用MIT许可证。

