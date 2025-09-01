# AI Chatbot 项目方案

## 1. 项目总目标、外部输入输出与实现功能

**项目总目标：**
开发一个AI Chatbot，它将具有以下核心功能：

1.  **记忆系统：**
    *   **长期记忆库：** 基于Milvus，存储原始文字和总结。每一句话都带着上文存储，并与上文一起编码。总结与原始文字需区分。
    *   **短期记忆库：** 基于内存，按时间顺序维护。

2.  **视觉模型功能：**
    *   支持U3D和VTubeStudio两个方案。
    *   以有限状态机的形式封装成状态转换接口。
    *   主模型通过function call形式调用转换接口以实现模型控制。

3.  **Grounding Agent功能：**
    *   利用基础模型的grounding能力。
    *   实现简单的grounding agent能力，例如通过视觉方案下五子棋和操作浏览器访问网页。

**外部输入：**
*   用户的文本输入（与Chatbot对话）。未来可能添加ASR输入，但最终也会作为文本处理。

**外部输出：**
*   Chatbot的文本回复。
*   视觉模型的控制指令（例如动画切换、表情变化等）。
*   Grounding Agent的操作指令（例如五子棋的落子指令、浏览器操作指令等）。

**实现功能：**
*   **基础模型集成：** 使用现有模型（例如Qwen2.5VL-api）作为基模，提供常识和基本理解能力。
*   **记忆管理：** 有效存储和检索长期和短期记忆，以支持连贯对话。
*   **视觉交互：** 根据对话内容和模型输出，控制虚拟形象的姿态和表情。
*   **具身交互：** 执行与物理世界或虚拟环境相关的任务，例如游戏和网页浏览。

## 2. 项目结构拆分

### 模块概述：

1.  **核心控制器 (Core Controller)**
    *   **功能：** 协调各个模块，管理整体对话流程，接收由输出模块处理后的文本回复，并将其发送给 TTS 等进一步处理（如果需要）。
    *   **技术栈：** Python

2.  **基础模型接口 (Base Model Interface)**
    *   **功能：** 负责与外部基础模型（如Qwen2.5VL-api）进行通信，发送由输入模块整理的用户输入和历史对话，接收模型输出（文本回复、function call等）。
    *   **技术栈：** Python，API客户端库。

3.  **输入模块 (Input Module)**
    *   **功能：**
        *   接收原始用户文本输入。
        *   **上下文管理：** 根据对话状态和预设规则，组合来自长期记忆和短期记忆的上下文模板。
        *   管理对长期记忆和短期记忆的 API 调用，以获取相关上下文。
        *   将整理后的用户输入和上下文发送给基础模型接口。
    *   **技术栈：** Python

4.  **输出模块 (Output Module)**
    *   **功能：**
        *   接收来自基础模型接口的原始输出。
        *   **结构化解析：** 识别并提取由特殊标记（例如 `<motion>动作</motion>` 或 ```json{动作}```）包装的结构化输出。
        *   将结构化输出解析为具体的行为指令，并分发给视觉交互模块和具身代理模块。
        *   将剩余的非结构化文本内容（例如，纯文本回复）输出，用于显示或发送给 TTS。
    *   **技术栈：** Python，正则表达式或JSON解析库。

5.  **记忆管理模块 (Memory Management Module)**
    *   **功能：**
        *   **长期记忆 (Long-term Memory)：** 存储原始文字和总结，支持语义检索。对外暴露 API 供输入模块调用。
        *   **短期记忆 (Short-term Memory)：** 存储当前对话上下文，维护时间顺序。对外暴露 API 供输入模块调用。
    *   **技术栈：** Python，Milvus（长期记忆），Python数据结构（如列表、字典用于短期记忆）。

6.  **视觉交互模块 (Visual Interaction Module)**
    *   **功能：**
        *   **U3D/VTubeStudio适配层：** 封装U3D和VTubeStudio的API，提供统一的状态转换接口。
        *   **状态机逻辑：** 根据输出模块的指令，执行虚拟形象的状态转换（例如表情、姿态、动画）。
    *   **技术栈：** Python，可能需要特定的SDK或网络通信（如WebSocket/TCP）与U3D/VTubeStudio实例交互。

7.  **具身代理模块 (Grounding Agent Module)**
    *   **功能：**
        *   **环境交互接口：** 提供与外部环境（如五子棋游戏、浏览器）交互的接口。
        *   **任务执行逻辑：** 根据输出模块的指令，执行具体的具身任务（例如五子棋落子、浏览器导航、点击等）。
    *   **技术栈：** Python，可能需要使用Selenium或其他自动化工具进行浏览器操作；对于五子棋游戏，可能需要自定义接口或游戏SDK。

### 模块间依赖关系（更新）：

*   **核心控制器** 协调高层逻辑。
*   **输入模块** 依赖于 **记忆管理模块**。
*   **基础模型接口** 接收来自 **输入模块** 的数据，并将输出发送给 **输出模块**。
*   **输出模块** 依赖于 **视觉交互模块** 和 **具身代理模块**。

## 3. 工作流

一次标准的工作流（例如，用户说一句话，Chatbot回应并可能伴随动作）如下：

1.  **用户输入**
    *   **接收：** 用户的原始文本输入。
    *   **做什么：：** 将文本输入传递给**输入模块**。
    *   **输出：** 用户文本输入。

2.  **输入模块 (Input Module)**
    *   **接收：** 用户的原始文本输入。
    *   **做什么：**
        *   根据对话状态和预设规则，调用**记忆管理模块**的API，检索相关短期记忆（当前对话历史）和长期记忆（相关知识、过去对话总结等）。
        *   将用户输入与检索到的上下文进行整合，形成一个完整的prompt。
    *   **输出：** 整合后的prompt（包含用户输入和相关上下文），发送给**基础模型接口**。

3.  **基础模型接口 (Base Model Interface)**
    *   **接收：** 整合后的prompt。
    *   **做什么：** 将prompt发送给外部的基础模型（如Qwen2.5VL-api），等待其响应。
    *   **输出：** 基础模型的原始输出，其中可能包含纯文本回复和由特殊标记（例如 `<motion>动作</motion>` 或 ```json{动作}```）包装的结构化动作指令。

4.  **输出模块 (Output Module)**
    *   **接收：** 来自**基础模型接口**的原始输出。
    *   **做什么：**
        *   **结构化解析：** 解析原始输出，识别并提取所有由特殊标记包装的结构化动作指令。
        *   将提取出的视觉动作指令分发给**视觉交互模块**。
        *   将提取出的具身代理动作指令分发给**具身代理模块**。
        *   提取原始输出中剩余的纯文本内容。
    *   **输出：**
        *   视觉动作指令发送给**视觉交互模块**。
        *   具身代理动作指令发送给**具身代理模块**。
        *   纯文本内容发送给**核心控制器**。

5.  **视觉交互模块 (Visual Interaction Module)**
    *   **接收：** 来自**输出模块**的视觉动作指令。
    *   **做什么：** 根据接收到的指令（例如“微笑”、“点头”），通过适配层调用U3D或VTubeStudio的API，执行虚拟形象的状态转换（如表情变化、姿态调整、动画播放）。
    *   **输出：** 虚拟形象的视觉表现更新。

6.  **具身代理模块 (Grounding Agent Module)**
    *   **接收：** 来自**输出模块**的具身代理动作指令。
    *   **做什么：** 根据接收到的指令（例如“下五子棋在E5”、“打开百度并搜索AI”），调用相应的环境交互接口（如Selenium进行浏览器操作，或五子棋游戏接口），执行具身任务。
    *   **输出：** 具身任务的执行结果（例如，浏览器页面变化，五子棋落子）。

7.  **核心控制器 (Core Controller)**
    *   **接收：** 来自**输出模块**的纯文本内容。
    *   **做什么：** 将纯文本内容进一步处理，例如发送给 TTS (Text-to-Speech) 模块进行语音合成，或直接显示给用户。同时，将当前轮次的对话内容（用户输入和Chatbot回复）发送给**记忆管理模块**进行存储和更新。
    *   **输出：** 最终的用户可见回复（文本或语音）。

8.  **记忆管理模块 (Memory Management Module)**
    *   **接收：**
        *   来自**输入模块**的上下文检索请求。
        *   来自**核心控制器**的对话内容更新请求。
    *   **做什么：**
        *   **长期记忆：** 根据检索请求进行语义匹配和检索，存储新的原始文字，并根据需要对对话进行总结并存储。确保每一句话都与上文一起编码。
        *   **短期记忆：** 更新当前对话历史，按时间顺序维护。
    *   **输出：** 上下文检索结果（返回给**输入模块**），或完成存储操作。

## 4. 各模块之间的交互

### 4.1. 核心控制器 (Core Controller)

*   **与输出模块 (Output Module) 交互：**
    *   **提供功能：** 无（核心控制器接收输出模块的文本）
    *   **接收功能：** `get_text_response(text: str)` - 接收处理后的纯文本回复。
    *   **传递数据：** `str` (纯文本回复)
    *   **API 格式示例：** `output_module.send_text_to_controller(text)`

*   **与记忆管理模块 (Memory Management Module) 交互：**
    *   **提供功能：** `store_dialogue(user_utterance: str, chatbot_response: str)` - 存储当前轮次的用户话语和Chatbot回复。
    *   **接收功能：** 无
    *   **传递数据：** `user_utterance: str`, `chatbot_response: str`
    *   **API 格式示例：** `memory_management_module.store_dialogue(user_text, chatbot_text)`

### 4.2. 输入模块 (Input Module)

*   **与记忆管理模块 (Memory Management Module) 交互：**
    *   **提供功能：** 无（输入模块调用记忆管理模块的功能）
    *   **接收功能：**
        *   `retrieve_short_term_memory() -> List[str]` - 获取短期记忆（当前对话历史）。
        *   `retrieve_long_term_memory(query: str, top_k: int) -> List[Tuple[str, float]]` - 根据查询检索长期记忆。
    *   **传递数据：**
        *   `query: str` (用于长期记忆检索的查询文本)
        *   `top_k: int` (长期记忆检索的数量)
        *   **返回数据：** `List[str]` (短期记忆列表)， `List[Tuple[str, float]]` (长期记忆及其相似度分数列表)。
    *   **API 格式示例：**
        *   `short_term_history = memory_management_module.retrieve_short_term_memory()`
        *   `long_term_results = memory_management_module.retrieve_long_term_memory(user_input, top_k=5)`

*   **与基础模型接口 (Base Model Interface) 交互：**
    *   **提供功能：** `send_prompt(prompt: str)` - 发送整合后的prompt给基础模型。
    *   **接收功能：** 无
    *   **传递数据：** `prompt: str` (整合后的用户输入和上下文)
    *   **API 格式示例：** `base_model_interface.send_prompt(combined_prompt)`

### 4.3. 基础模型接口 (Base Model Interface)

*   **与输入模块 (Input Module) 交互：**
    *   **提供功能：** 无（基础模型接口接收输入模块的prompt）
    *   **接收功能：** `get_prompt(prompt: str)` - 接收来自输入模块的prompt。
    *   **传递数据：** `prompt: str`
    *   **API 格式示例：** `input_module.send_prompt(prompt)`

*   **与输出模块 (Output Module) 交互：**
    *   **提供功能：** `send_raw_output(raw_output: str)` - 发送基础模型的原始输出。
    *   **接收功能：** 无
    *   **传递数据：** `raw_output: str` (基础模型的原始输出，包含文本和结构化指令)
    *   **API 格式示例：** `output_module.process_raw_model_output(model_raw_output)`

### 4.4. 输出模块 (Output Module)

*   **与基础模型接口 (Base Model Interface) 交互：**
    *   **提供功能：** 无（输出模块接收基础模型接口的原始输出）
    *   **接收功能：** `get_raw_output(raw_output: str)` - 接收来自基础模型接口的原始输出。
    *   **传递数据：** `raw_output: str`
    *   **API 格式示例：** `base_model_interface.send_raw_output(model_output)`

*   **与视觉交互模块 (Visual Interaction Module) 交互：**
    *   **提供功能：** `execute_visual_action(action_data: Dict)` - 执行视觉动作。
    *   **接收功能：** 无
    *   **传递数据：** `action_data: Dict` (包含视觉动作类型、参数等，例如 `{"type": "expression", "name": "smile", "intensity": 0.8}`)
    *   **API 格式示例：** `visual_interaction_module.execute_visual_action({"type": "motion", "name": "wave", "duration": 2.0})`

*   **与具身代理模块 (Grounding Agent Module) 交互：**
    *   **提供功能：** `execute_grounding_action(action_data: Dict)` - 执行具身代理动作。
    *   **接收功能：** 无
    *   **传递数据：** `action_data: Dict` (包含具身动作类型、参数等，例如 `{"type": "gobang_move", "position": "E5"}` 或 `{"type": "browser_open", "url": "https://www.google.com"}`)
    *   **API 格式示例：** `grounding_agent_module.execute_grounding_action({"type": "browser_navigate", "url": "https://www.example.com"})`

*   **与核心控制器 (Core Controller) 交互：**
    *   **提供功能：** `send_text_response(text: str)` - 发送纯文本回复。
    *   **接收功能：** 无
    *   **传递数据：** `text: str` (纯文本回复)
    *   **API 格式示例：** `core_controller.receive_text_response(plain_text_output)`

### 4.5. 记忆管理模块 (Memory Management Module)

*   **与输入模块 (Input Module) 交互：**
    *   **提供功能：**
        *   `retrieve_short_term_memory() -> List[str]`
        *   `retrieve_long_term_memory(query: str, top_k: int) -> List[Tuple[str, float]]`
    *   **接收功能：** 无
    *   **传递数据：**
        *   **接收：** `query: str`, `top_k: int`
        *   **返回：** `List[str]`, `List[Tuple[str, float]]`
    *   **API 格式示例：** `input_module` 调用 `memory_management_module` 的相应方法。

*   **与核心控制器 (Core Controller) 交互：**
    *   **提供功能：** `store_dialogue(user_utterance: str, chatbot_response: str)`
    *   **接收功能：** 无
    *   **传递数据：** `user_utterance: str`, `chatbot_response: str`
    *   **API 格式示例：** `core_controller` 调用 `memory_management_module.store_dialogue`。

### 4.6. 视觉交互模块 (Visual Interaction Module)

*   **与输出模块 (Output Module) 交互：**
    *   **提供功能：** 无（视觉交互模块接收输出模块的指令）
    *   **接收功能：** `execute_visual_action(action_data: Dict)` - 接收来自输出模块的视觉动作指令。
    *   **传递数据：** `action_data: Dict`
    *   **API 格式示例：** `output_module` 调用 `visual_interaction_module.execute_visual_action`。

### 4.7. 具身代理模块 (Grounding Agent Module)

*   **与输出模块 (Output Module) 交互：**
    *   **提供功能：** 无（具身代理模块接收输出模块的指令）
    *   **接收功能：** `execute_grounding_action(action_data: Dict)` - 接收来自输出模块的具身代理动作指令。
    *   **传递数据：** `action_data: Dict`
    *   **API 格式示例：** `output_module` 调用 `grounding_agent_module.execute_grounding_action`.

## 5. 开发流程

以下是建议的开发流程，旨在逐步构建和集成各个模块：

1.  **基础设施准备 (Phase 1: Infrastructure Setup)**
    *   **目标：** 搭建核心服务和开发环境。
    *   **步骤：**
        *   **Milvus安装与配置：** 安装和配置Milvus向量数据库，确保其可访问。
        *   **基础模型API接入准备：** 获取Qwen2.5VL-api等基础模型的API密钥和访问凭证，并进行初步测试以确认连接正常。
    *   **测试与验证：** 确认Milvus服务正常运行，基础模型API能够成功调用并返回响应。

2.  **核心模块开发 (Phase 2: Core Module Development)**
    *   **目标：** 实现记忆管理和基础模型通信的核心功能。
    *   **步骤：**
        *   **记忆管理模块 (Memory Management Module)：**
            *   实现短期记忆的内存存储和时间顺序维护逻辑。
            *   实现长期记忆与Milvus的集成，包括文本嵌入、存储原始文字和总结、以及语义检索功能。
            *   定义并实现供输入模块和核心控制器调用的API。
        *   **基础模型接口 (Base Model Interface)：**
            *   实现与Qwen2.5VL-api等基础模型的请求发送和响应接收逻辑。
            *   处理API调用的错误和重试机制。
    *   **测试与验证：**
        *   单元测试：验证记忆管理模块的存储、检索功能和API正确性。
        *   单元测试：验证基础模型接口能够正确发送prompt并接收/解析响应。

3.  **输入/输出模块开发 (Phase 3: Input/Output Module Development)**
    *   **目标：** 实现用户输入处理和模型输出解析与分发。
    *   **步骤：**
        *   **输入模块 (Input Module)：**
            *   实现接收用户原始文本输入的功能。
            *   开发上下文管理逻辑，包括模块化上下文模板的加载和应用规则。
            *   集成记忆管理模块的API，以获取短期和长期记忆，并将其整合到发送给基础模型的prompt中。
        *   **输出模块 (Output Module)：**
            *   实现接收来自基础模型接口的原始输出。
            *   开发结构化解析逻辑，识别并提取 `<motion>动作</motion>` 或 ```json{动作}``` 等特殊标记的结构化指令。
            *   实现将解析后的动作指令分发给视觉交互模块和具身代理模块的功能。
            *   实现提取并传递纯文本内容给核心控制器的功能。
    *   **测试与验证：**
        *   单元测试：验证输入模块的上下文生成和prompt构建的正确性。
        *   单元测试：验证输出模块的结构化解析和指令分发逻辑。
        *   集成测试：验证输入模块、记忆管理模块和基础模型接口之间的端到端通信。

4.  **接口适配层开发 (Phase 4: Interface Adaptation Layer Development)**
    *   **目标：** 实现与U3D/VTubeStudio和外部环境的通信接口。
    *   **步骤：**
        *   **视觉交互模块 (Visual Interaction Module)：**
            *   开发U3D和VTubeStudio的适配层，封装各自的API，提供统一的状态转换接口。
            *   实现状态机逻辑，根据接收到的指令执行虚拟形象的表情、姿态、动画等控制。
        *   **具身代理模块 (Grounding Agent Module) - 环境交互接口部分：**
            *   开发与外部环境（例如，浏览器、五子棋游戏）交互的抽象接口。
            *   为浏览器操作实现基于Selenium或其他自动化工具的接口。
    *   **测试与验证：**
        *   单元测试：验证视觉交互模块能够正确控制U3D/VTubeStudio虚拟形象的动作。
        *   单元测试：验证具身代理模块的环境交互接口能够执行基本操作（例如，浏览器打开网页）。

5.  **核心控制器开发 (Phase 5: Core Controller Development)**
    *   **目标：** 整合所有模块，实现Chatbot的整体对话流程。
    *   **步骤：**
        *   实现主循环，接收用户输入，调用输入模块，等待输出模块响应，并将文本输出交给TTS或显示。
        *   协调各模块之间的调用顺序和数据流。
        *   实现将对话内容发送给记忆管理模块进行存储的逻辑。
    *   **测试与验证：**
        *   集成测试：进行完整的端到端测试，验证Chatbot从接收用户输入到输出响应（包括视觉和具身动作）的整个流程。

6.  **具身代理功能实现 (Phase 6: Grounding Agent Functionality)**
    *   **目标：** 实现具体的具身代理任务。
    *   **步骤：**
        *   **五子棋功能：** 在具身代理模块中实现五子棋游戏的交互逻辑（例如，接收落子指令，更新棋盘状态，反馈结果）。
        *   **网页访问操作：** 扩展浏览器操作接口，实现更复杂的网页访问、信息提取和交互功能。
    *   **测试与验证：** 功能测试：验证五子棋和网页访问功能的正确性和稳定性。

7.  **集成测试与系统验证 (Phase 7: Integration and System Validation)**
    *   **目标：** 确保整个系统稳定、高效，并满足所有功能需求。
    *   **步骤：**
        *   **全面集成测试：** 对所有模块进行全面集成测试，发现并修复潜在的兼容性问题。
        *   **性能测试：** 评估系统在不同负载下的响应时间、内存占用等性能指标。
        *   **用户体验测试：** 进行用户验收测试，收集用户反馈并进行优化。
    *   **测试与验证：** 完整系统测试报告，性能基线。

8.  **数据准备与模型微调 (Phase 8: Data Preparation and Model Fine-tuning - Optional)**
    *   **目标：** 如果需要提升特定领域的表现或增加特定行为，进行数据准备和模型微调。
    *   **步骤：**
        *   **数据收集与标注：** 收集与特定任务或领域相关的数据，并进行标注。
        *   **模型微调：** 使用标注数据对基础模型进行微调（如果基础模型支持）。
    *   **测试与验证：** 微调模型在特定任务上的表现评估.
