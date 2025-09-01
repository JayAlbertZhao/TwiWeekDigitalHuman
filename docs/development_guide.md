# AI Chatbot 开发指南

## 1. 项目核心功能

该AI Chatbot将具备以下核心功能：

*   **记忆系统：**
    *   **长期记忆库：** 基于Milvus，存储原始文字和总结，每句话带上文编码，区分原始文字与总结。
    *   **短期记忆库：** 基于内存实现，维护当前对话上下文，通常不超过200句或5k tokens，以方便、高效为主要考量。
*   **虚拟形象控制模块 (Virtual Avatar Control Module)：** 支持U3D和VTubeStudio，以有限状态机形式提供状态转换接口，主模型通过function call控制虚拟形象。
*   **Grounding Agent功能：** 利用基础模型（如Qwen2.5VL-api）的grounding能力，实现通过视觉下五子棋和操作浏览器访问网页等具身任务。

## 2. 模块结构

项目包含以下主要模块及其职责：

1.  **核心控制器 (Core Controller)：** 协调各模块，管理对话流程，接收处理后的文本输出。
2.  **基础模型接口 (Base Model Interface)：** 与外部基础模型（如Qwen2.5VL-api）通信，发送prompt，接收原始输出。
3.  **输入模块 (Input Module)：** 接收用户文本，**内部包含上下文管理系统**，负责整合长短期记忆，构建并发送prompt。
4.  **输出模块 (Output Module)：** 解析基础模型原始输出，提取结构化指令（视觉/具身动作），分发给相应模块，并传递纯文本回复。
5.  **记忆管理模块 (Memory Management Module)：** 提供长期记忆（Milvus）和短期记忆（内存）的存储、检索API，供输入模块的上下文管理系统调用。
6.  **虚拟形象控制模块 (Virtual Avatar Control Module)：** 适配U3D/VTubeStudio，实现状态机逻辑，执行虚拟形象的视觉动作。
7.  **具身代理模块 (Grounding Agent Module)：** 提供与外部环境（如五子棋、浏览器）交互的接口，执行具身任务。
8.  **TTS模块 (Text-to-Speech Module)：** 接收纯文本，将其转换为语音输出。

## 3. 工作流概述

一次标准交互流程：

1.  **用户输入** -> **输入模块** (**上下文管理系统**获取上下文, 构建prompt)
2.  **输入模块** -> **基础模型接口** (发送prompt)
3.  **基础模型接口** -> **输出模块** (接收原始输出)
4.  **输出模块** (解析输出, 分发指令)
    *   结构化指令 -> **虚拟形象控制模块** (执行视觉动作)
    *   结构化指令 -> **具身代理模块** (执行具身任务)
    *   纯文本回复 -> **核心控制器** (处理显示) -> **TTS模块** (语音输出)
5.  **核心控制器** -> **记忆管理模块** (更新对话记忆)

## 4. 模块间交互与API（示例）

*   **输入模块 (上下文管理系统) <-> 记忆管理模块：**
    *   `retrieve_short_term_memory() -> List[str]`
    *   `retrieve_long_term_memory(query: str, top_k: int) -> List[Tuple[str, float]]`
*   **输入模块 -> 基础模型接口：**
    *   `send_prompt(prompt: str)`
*   **基础模型接口 -> 输出模块：**
    *   `send_raw_output(raw_output: str)`
*   **输出模块 -> 虚拟形象控制模块：**
    *   `execute_avatar_action(action_data: Dict)` (e.g., `{"type": "expression", "name": "smile"}`)
*   **输出模块 -> 具身代理模块：**
    *   `execute_grounding_action(action_data: Dict)` (e.g., `{"type": "gobang_move", "position": "E5"}`)
*   **输出模块 -> 核心控制器：**
    *   `send_text_response(text: str)`
*   **核心控制器 -> 记忆管理模块：**
    *   `store_dialogue(user_utterance: str, chatbot_response: str)`
*   **核心控制器 -> TTS模块：**
    *   `convert_text_to_speech(text: str) -> AudioData`

### 4.8. API 兼容性设计原则

为了实现不同基础模型 (LLM)、TTS服务和虚拟形象控制方案的灵活切换，我们将遵循类似 `transformers` 库的接口设计原则：

*   **统一接口定义：** 定义一套标准化的抽象接口（如`LLMInterface`, `TTSInterface`, `AvatarControlInterface`），所有兼容的模型/服务都必须实现这些接口。
*   **多实现后端：** 不同的具体模型或服务将作为这些接口的独立实现。例如，`Qwen25VLAPI` 和 `AnotherLLMAPI` 都实现 `LLMInterface`。
*   **动态加载/配置：** 系统应支持通过配置文件或运行时参数，动态选择并加载特定的后端实现。

**示例：**

```python
# LLM接口示例
class LLMInterface:
    def generate(self, prompt: str, context: List[str]) -> Dict:
        raise NotImplementedError

class Qwen25VLAPI(LLMInterface):
    def generate(self, prompt: str, context: List[str]) -> Dict:
        # ... 调用 Qwen2.5VL-api 的具体实现 ...
        pass

# TTS接口示例
class TTSInterface:
    def synthesize(self, text: str) -> AudioData:
        raise NotImplementedError

class CustomTTS(TTSInterface):
    def synthesize(self, text: str) -> AudioData:
        # ... 调用自定义 TTS 服务的具体实现 ...
        pass

# 虚拟形象控制接口示例
class AvatarControlInterface:
    def execute_action(self, action_data: Dict):
        raise NotImplementedError

class U3DControl(AvatarControlInterface):
    def execute_action(self, action_data: Dict):
        # ... 与 U3D 交互的具体实现 ...
        pass
```

## 5. 开发流程

建议的开发顺序为：

1.  **基础设施准备：** Milvus安装配置，基础模型API接入测试。
2.  **核心模块开发：** 实现记忆管理模块和基础模型接口的核心功能及API。
3.  **输入/输出模块开发：** 实现输入模块的上下文管理、prompt构建，以及输出模块的结构化解析和指令分发。
4.  **接口适配层开发：** 实现**虚拟形象控制模块**的U3D/VTubeStudio适配层及状态机，以及具身代理模块的环境交互接口（如浏览器自动化）。
5.  **核心控制器开发：** 整合所有模块，实现对话主循环和数据流，更新记忆。
6.  **具身代理功能实现：** 具体实现五子棋、网页访问等具身任务。
7.  **TTS模块开发：** 实现文本到语音转换功能。
8.  **集成测试与系统验证：** 全面集成、性能和用户体验测试。
9.  **数据准备与模型微调 (可选)：** 特定领域数据收集、标注及模型微调。
