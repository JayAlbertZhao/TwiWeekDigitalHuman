# 输入模块 (Input Module) IO 文档

## 1. 模块概述

输入模块负责接收用户的原始文本输入，并通过其内部的上下文管理系统，整合长短期记忆，构建发送给基础模型（LLM）的完整prompt。其核心功能在于灵活地管理和应用上下文模板，以适应不同的对话场景和需求。

## 2. `InputModule` 类定义

`InputModule` 应设计为一个类，其初始化方法 (`__init__`) 将负责加载和解析配置信息。配置信息可以是一个JSON文件，它定义了上下文模板和其他相关设置。

### 2.1. `__init__` 方法

`InputModule` 的 `__init__` 方法应负责以下任务：

1.  接收一个可选的 `config_path` 参数，指向JSON配置文件。
2.  如果 `config_path` 为空或无效，则加载默认配置。
3.  解析加载的配置，特别是其中的上下文模板 (`context_template`)。
4.  初始化必要的内部状态和资源。

## 3. Config 文件结构

`InputModule` 的JSON配置文件 (`config_path`指向的文件) 应包含以下主要结构：

```json
{
    "context_template": [
        {
            "module_name": "module_A",
            "segments": [
                {"type": "text", "value": "固定文本段1"},
                {"type": "variable", "value": "variable_name_1"}
            ],
            "importance_func": true 
        },
        {
            "module_name": "module_B",
            "segments": [
                {"type": "text", "value": "固定文本段2"},
                {"type": "variable", "value": "function_name_1"}
            ],
            "importance_func": "is_relevant_func" 
        }
    ]
    // ... 其他内容定义 ...
}
```

### 3.1. `context_template`

这是一个有序的模块列表，直接定义了上下文的组成顺序。不再使用模板名称作为键。

### 3.2. 上下文模板中的模块 (Module)

每个模块是一个字典，包含以下字段：

*   `"module_name"` (string, **必需**): 模块的唯一标识符（例如 `"short_term_memory"`, `"user_input"`）。
*   `"segments"` (list, **必需**): 一个有序的段 (segment) 列表，每个段可以是固定文本或变量。
*   `"importance_func"` (boolean or string, **可选**, 默认为 `true`): 
    *   如果为 `true`，则该模块总是被包含在最终的prompt中。
    *   如果为 `false`，则该模块总是被排除。
    *   如果是一个字符串，则该字符串应对应 `InputModule` 内部的一个方法名 (例如 `"is_relevant_func"`)，该方法将根据当前对话状态和检索结果动态判断是否包含该模块。该方法应返回 `True` 或 `False`。

### 3.3. 模块中的段 (Segment)

每个段是一个字典，包含以下字段：

*   `"type"` (string, **必需**): 段的类型，可以是 `"text"` 或 `"variable"`。
*   `"value"` (string, **必需**): 
    *   如果 `"type"` 为 `"text"`，则 `"value"` 是要插入的固定文本。
    *   如果 `"type"` 为 `"variable"`，则 `"value"` 是一个变量名，其值将由 `InputModule` 在构建prompt时动态获取（例如，从记忆管理模块获取的短时记忆内容）。如果 `"value"` 是一个方法名 (例如 `"function_name_1"`)，则 `InputModule` 内部需要调用该方法来获取变量值。

## 4. 变量来源和解析

`"variable"` 类型段中的 `"value"` 字段将对应于 `InputModule` 运行时需要填充的数据。这些数据通常来源于：

*   **记忆管理模块：** 例如，`"short_term_history_content"` 变量可能对应 `memory_management_module.retrieve_short_term_memory()` 的结果，经过格式化后插入。
*   **当前用户输入：** 例如，`"current_user_input"` 直接对应用户当前键入的文本。
*   **重要性函数或变量获取函数：** `InputModule` 内部需要定义能够根据 `"value"` 字符串动态调用相应函数来获取数据或判断重要性的机制。这些函数可以是 `InputModule` 的成员方法，或者通过其他方式注册。
*   **其他内部状态：** 可能包含 `InputModule` 内部维护的其他状态信息。

`InputModule` 内部需要实现一个机制，能够根据 `"variable"` 的 `"value"` 来获取并格式化相应的数据。对于短期记忆库，由于其规模相对较小 (不超过200句或5k tokens)，可以采用Python内置的数据结构（如列表或双端队列 `collections.deque`）来实现，无需引入复杂的数据库方案。
