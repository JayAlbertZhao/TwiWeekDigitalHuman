import asyncio
from ABCs import AsyncModule
from avatar import create_avatar_module, AvatarModule
from milvus_database import create_memory_module, MemoryModule

class CoreModule():
    def __init__(self):
        self._is_ready = asyncio.Event()
        self.all_modules = []
        self.avatar_module = None
        self.memory_module = None
        self.shutdown_event = asyncio.Event()
        self.ongoing_message = None

    async def _setup(self):
        if not self._is_ready.is_set():
            self.avatar_module = await create_avatar_module()
            self.all_modules.append(self.avatar_module)
            self.memory_module = await create_memory_module()
            self.all_modules.append(self.memory_module)
            self._is_ready.set()

    async def shutdown(self):
        for module in self.all_modules:
            await module.shutdown()
    
    async def main_loop(self):
        while not self.shutdown_event.is_set():
            """
            如果UI返回新的消息，检查当前消息是否已经处理完成（是否未清除）
            if self.ongoing_message is None:
                self.ongoing_message = [self.ui_message]
                开始回复当前消息：
                    （可以）模型进入思考/聆听姿势
                    传入prompt模板模块
                    对每个模板组件
                        查询模板匹配
                            如果匹配成功，进行模板查询
                            否则进行常规查询
                        根据查询结果组成prompt组件
                    组合成完整prompt
                    传入基础模型接口
                    解析模型返回
                        动作指令传入avatar模块
                        行为指令传入agent操作模块
                        思考性内容传入ui模块详细内容
                        剩余内容传入tts模块生成语音
                        剩余内容与语音传输给ui模块展示
            else:
                终止当前回复
                self.ongoing_message.append(self.ui_message)
                开始回复当前消息
            """
            await asyncio.sleep(1)