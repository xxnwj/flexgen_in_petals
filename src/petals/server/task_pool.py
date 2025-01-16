import ctypes
import multiprocessing as mp
import threading
import time
from concurrent.futures._base import PENDING
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from hivemind import get_logger
from hivemind.utils.mpfuture import ALL_STATES, MPFuture

logger = get_logger(__name__)


@dataclass(order=True, frozen=True)
class Task:
    priority: float
    time_submitted: float
    future: MPFuture = field(compare=False)
    args: Sequence[torch.Tensor] = field(compare=False)

    @property
    def uid(self) -> int:
        return self.future._uid


class PrioritizedTaskPool(threading.Thread):
    """
    Aggregates requests from multiple ConnectionHandler instances, orders them for processing in Runtime, then
    returns results (or exception) to the corresponding ConnectionHandler. Runs a background process.
    A single PrioritizedTaskPool services a specific function (e.g. layer1.forward, layer2.forward or layer1.backward)

    :note: unlike hivemind.moe TaskPool, this pool does *not* combine incoming requests into batches.
      This would require grouping requests of different length.

    :param process_func: function to be applied to every formed batch; called by Runtime
        Note that process_func should accept only positional args (Tensors) and return a flat tuple of Tensors
    :param max_batch_size: process at most this many inputs in a batch (task contains have one or several inputs)
         Measured in the total number of tokens (i.e. batch size * sequence length)

    :param name: pool name, used for logging
    :param min_batch_size: process at least this many inputs in a batch, otherwise wait for more
    :param device: if specified, input tensors will be moved to that device by default
    :param start: if True, start automatically at the end of __init__
    """

    def __init__(
        self,
        process_func: callable,
        max_batch_size: int,
        name: str,
        min_batch_size=1,
        device: Optional[torch.device] = None,
        daemon=True,
        start=False,
    ):
        super().__init__(daemon=daemon, name=name)
        self.process_func = process_func
        # the lower the priority is, the more urgent it is to process this pool
        self._priority = mp.Value(ctypes.c_double, 1.0)

        self.min_batch_size, self.max_batch_size = min_batch_size, max_batch_size
        self.device = device

        self.submitted_tasks = mp.SimpleQueue()  # interaction with ConnectionHandlers
        self._ordered_tasks = PriorityQueue()  # interaction with Runtime - only valid inside Runtime

        self._dispatched_tasks = {}
        self.batch_receiver, self.batch_sender = mp.Pipe(duplex=False)
        self._oldest_undispatched_timestamp = mp.Value(ctypes.c_double, 1.0)
        self.priority = float("inf"), float("inf")  # (first task priority, first task timestamp)

        if start:
            self.start()

    def run(self):
        """Read tasks from incoming queue and put them into a local priority queue"""
        while True:
            task = self.submitted_tasks.get()
            if task is None:
                logger.debug("Shutting down prioritizer thread")
                break

            self._ordered_tasks.put(task, block=True) # _ordered_tasks is a PriorityQueue

    def terminate(self):
        """An alias for hivemind.Runtime that assumes that each TaskPool is a process"""
        self.shutdown()

    def shutdown(self):
        self.submitted_tasks.put(None)  # Shuts down self.run()

    def submit_task(self, *args: Any, priority: float = 0.0, seg_lengths: Optional[List[int]] = None) -> MPFuture:
        """ Add task to this pool's queue, return Future for its output.
            主要用于管理任务的提交，确保任务在处理前符合大小限制，并维护任务的优先级。
            通过 MPFuture 对象，调用者可以在任务完成时获取结果或处理异常。
        """
        future = MPFuture()
        # Remove shared memory from MPFuture. This disables the .cancel() feature but
        # saves the server from "could not unlink the shared memory file" crashes during rebalancing
        future._shared_state_code = torch.tensor([ALL_STATES.index(PENDING)], dtype=torch.uint8)

        task = Task(priority, time.monotonic(), future, args)# 创建一个任务对象，包含优先级、提交时间、未来对象和任务参数 
        if self.get_task_size(task) > self.max_batch_size: # 检查任务的大小是否超过最大批处理大小  
            exc = ValueError(f"Task size greater than max_batch_size ({self.max_batch_size}), it can't be processed")# 如果超过，抛出异常
            task.future.set_exception(exc) # 将异常设置到未来对象中  
        else:
            self.submitted_tasks.put(task) # 将任务添加到提交的任务队列中, submitted_tasks is a SimpleQueue
            self.batch_sender.send(None)  # use this pipe to count the number of unfinished batches # 使用此管道来计算未完成批次的数量  
            if (task.priority, task.time_submitted) < self.priority: # 检查新任务的优先级和提交时间
                self.priority = (task.priority, task.time_submitted) # 更新当前优先级
        return task.future # 返回任务的未来对象

    def get_task_size(self, task: Task) -> int:
        """compute task processing complexity; defaults to the total number of tokens"""
        if task.args and task.args[0].ndim >= 2:
            return task.args[0].shape[0] * task.args[0].shape[1]
        return 1

    def load_batch_to_runtime(
        self, timeout: Optional[float] = None, device: Optional[torch.device] = None
    ) -> Tuple[Any, List[torch.Tensor]]:
        """receive next batch of arrays"""
        device = device if device is not None else self.device # 如果未指定设备，则使用默认设备 
        print('-=-==-=-=-=-=- task pool: load_batch_to_runtime(): device ', device)
        task = self._ordered_tasks.get(block=True, timeout=timeout) # 从有序任务队列中获取下一个任务，可能会阻塞直到超时 
        batch_inputs = [_move_to_device_if_tensor(arg, device, share_memory=False) for arg in task.args] # 将任务参数移动到指定设备 
        self._dispatched_tasks[task.uid] = task # 将任务标记为已分派 
        self.batch_receiver.recv()  # reduce the number of active batches
        if not self._ordered_tasks.empty(): # 如果还有剩余任务  
            first_remaining_task: Task = self._ordered_tasks.queue[0] # 获取队列中的第一个剩余任务
            self.priority = (first_remaining_task.priority, first_remaining_task.time_submitted) # 更新当前优先级
        return task.uid, batch_inputs # 返回任务的唯一标识符和批次输入  

    def send_outputs_from_runtime(self, uid: int, batch_outputs: List[torch.Tensor]):
        """send results for a processed batch, previously loaded through load_batch_to_runtime"""
        batch_outputs = [_move_to_device_if_tensor(output, device="cpu", share_memory=True) for output in batch_outputs]
        task = self._dispatched_tasks.pop(uid, None)
        if task is None:
            logger.error(
                f"Internal error: task task with index {uid} is missing from the dictionary; " f"Could not set result"
            )
        else:
            task.future.set_result(batch_outputs)

    def send_exception_from_runtime(self, uid: int, exception: BaseException):
        task = self._dispatched_tasks.pop(uid, None)
        if task is None:
            logger.error(
                f"Internal error: task task with index {uid} is missing from the dictionary; "
                f"Could not set exception {exception}"
            )
        else:
            task.future.set_exception(exception)

    @property
    def empty(self):
        return not self.batch_receiver.poll()

    @property
    def priority(self) -> Tuple[float, float]:
        """The priority of this pool equals the (priority, timestamp) of the most important task in it."""
        return float(self._priority.value), float(self._oldest_undispatched_timestamp.value)

    @priority.setter
    def priority(self, item: Tuple[float, float]):
        assert len(item) == 2
        self._priority.value = float(item[0])
        self._oldest_undispatched_timestamp.value = float(item[1])


def _move_to_device_if_tensor(arg: Any, device: Union[torch.device, str], share_memory: bool = False):
    if isinstance(arg, torch.Tensor):
        arg = arg.detach().to(device, non_blocking=not share_memory).requires_grad_(arg.requires_grad)
        # note: it is important that non_blocking is disabled if share_memory=True; using share_memory on a tensor
        # produced by a non-blocking copy will result in undefined behavior (depending on your gpu speed)
        if share_memory:
            arg = arg.share_memory_()
    return arg
