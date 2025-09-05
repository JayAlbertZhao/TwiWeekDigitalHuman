import pyvts
import asyncio
from ABCs import AsyncModule

class AvatarModule(AsyncModule):
    def __init__(self):
        super().__init__()
        self.vts = pyvts.vts()
        self.task_queue = asyncio.Queue()
        self.hotkey_list = []
        self._shutdown = asyncio.Event()

    async def _setup(self):
        if not self._is_ready.is_set():
            await self.connect_auth(self.vts)
            await self.get_hotkey_list(self.vts)
            self._is_ready.set()

    async def connect_auth(self, myvts):
        await myvts.connect()
        await myvts.request_authenticate_token()
        await myvts.request_authenticate()

    async def get_hotkey_list(self, myvts):
        response_data = await myvts.request(myvts.vts_request.requestHotKeyList())
        print(response_data)
        self.hotkey_list = []
        for hotkey in response_data["data"]["availableHotkeys"]:
            self.hotkey_list.append(hotkey["name"])
        return self.hotkey_list
    
    async def enqueue_task(self, motion_name: str):
        self.task_queue.put_nowait(motion_name)
    
    async def process_task(self):
        while not self._shutdown.is_set():
            motion_name = await self.task_queue.get()
            await self.vts.request(self.vts.vts_request.requestTriggerHotKey(motion_name))
    
    async def shutdown(self):
        self._shutdown.set()
        await self.task_queue.join()
        await self.vts.close()

async def create_avatar_module() -> AvatarModule:
    """
    异步工厂函数，负责创建和初始化 AvatarModule 实例。
    """
    module = AvatarModule()
    # 异步地执行内部的设置方法
    setup_task = asyncio.create_task(module._setup())
    # 等待设置完成
    await module._is_ready.wait()
    return module
