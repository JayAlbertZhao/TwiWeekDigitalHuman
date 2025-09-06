import abc
from abc import ABC, abstractmethod
import asyncio
from typing import Any

class AsyncModule(ABC):
    """异步工作模块的抽象基类
    特征：工作是异步的，没有直接的返回值，运行即效果，可能需要等待时间
    """
    def __init__(self):
        self._is_ready = asyncio.Event()
        self._task_queue = asyncio.Queue()
        pass

    @abstractmethod
    async def _setup(self):
        """
        初始化模块时需要时间进行的操作放_setup而不放__init__，由factory调用
        示例：
        if not self._is_ready.is_set():
            # do something
            self._is_ready.set()
        """
        pass
    
    @abstractmethod
    async def enqueue_task(self, task: Any) -> None:
        """
        将任务加入队列，task处传任务参数，异步处理
        任务一般直接产生结果，无返回值
        示例：
        self._task_queue.put_nowait(task)
        """
        pass
    
    @abstractmethod
    async def process_task(self) -> None:
        """
        开启处理任务的循环，具体任务处理放在_process_task，或如果任务简单则直接在这里实现
        示例：
        while not self._task_queue.empty():
            task = await self._task_queue.get()
            await self._process_task(task)
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        关闭模块
        """
        pass

class InstantModule(ABC):
    """即时工作模块的抽象基类
    特征：工作是即时的，有直接的返回值，阻塞式运行
    """
    def __init__(self):
        self._is_ready = asyncio.Event()
        pass
    
    @abstractmethod
    async def _setup(self):
        """
        初始化模块时需要时间进行的操作放_setup而不放__init__，由factory调用
        示例：
        if not self._is_ready.is_set():
            # do something
            self._is_ready.set()
        """
        pass
    
    @abstractmethod
    async def shutdown(self):
        """
        关闭模块
        """
        pass