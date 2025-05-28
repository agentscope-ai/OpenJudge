from abc import ABC
from rm_gallery.core.data.base import BaseData


from abc import abstractmethod
from enum import IntEnum
from typing import Any, TypeVar, List, Callable

from pydantic import BaseModel, Field

from _del.executor.queue import ThreadSafeQueue

T = TypeVar("T", bound="BaseModule")

class RuntimeStatus(IntEnum):
    INIT: 0
    READY: 1
    RUNNING: 2
    WAIT: 3
    DONE: 4
    ERROR: 5

class BaseModule(BaseModel):

    @abstractmethod
    def run(self, **kwargs) -> Any:
        ...

    # @classmethod
    # def from_gallery(
    #         cls: Type[T],
    #         path: str,
    #         marshaller: Marshaller = DEFAULT_MARSHALLER,
    #         **kwargs) -> T:
    #     fp = file.load_from_gallery(path)
    #     return cls.from_dict(marshaller.unmarshal(fp.read()))

    def with_dependencies(self, denpendencies: List[str]) -> T:
        if isinstance(self, RuntimeModule):
            self.dependencies = denpendencies
            return self
        else:
            runtime_module = self.with_runtime_info(dependencies=denpendencies)
            return runtime_module

    def with_retry(self, max_retry_cnt: int) -> T:
        if isinstance(self, RuntimeModule):
            self.retry_max_cnt = max_retry_cnt
            return self
        else:
            runtime_module = self.with_runtime_info(max_retry_cnt = max_retry_cnt)
            return runtime_module

    def with_check_in(self, call_func:Callable[[Any],bool], **kwargs) -> T:
        if isinstance(self,RuntimeModule):
            self.check_in_func = call_func
            return self
        else:
            runtime_module = self.with_runtime_info(check_in_func=call_func)
            return runtime_module

    def with_runtime_name(self,runtime_name:str=None):
        if isinstance(self,RuntimeModule):
            self.runtime_name = runtime_name
        else:
            if not runtime_name:
                runtime_name = self.__class__.__name__
            runtime_module = self._with_runtime_info(run_time_name=runtime_name)
            return runtime_module


    def _with_runtime_info(self,
                          run_time_name:str = None,
                          dependencies : list[str] = None,
                          check_in_func : Callable[[Any],bool] = None,
                          retry_max_cnt: int = 1,
                          runtime_status: RuntimeStatus = RuntimeStatus.INIT,
                          timeout_seconds: int = 120
                          ):
        return RuntimeModule(bound=self,run_time_name=run_time_name,dependencies=dependencies,check_in_func=check_in_func,retry_max_cnt=retry_max_cnt,runtime_status=runtime_status,timeout_seconds=timeout_seconds)


class RuntimeModule(BaseModule):
    bound: BaseModule = Field(default=None)
    runtime_name: str = Field(default="module_default_name")
    dependencies: List[str] = Field(default=[])
    check_in_func: Callable[[Any],bool] = Field(default=None)
    retry_max_cnt: int = Field(default=1)
    runtime_status: RuntimeStatus = Field(default=RuntimeStatus.INIT)
    timeout_seconds: int = Field(default=120000)

    def __init__(self,
                  bound: BaseModule = None,
                  run_time_name:str = None,
                  dependencies : list[str] = None,
                  check_in_func : Callable[[Any],bool] = None,
                  retry_max_cnt: int = 1,
                  runtime_status: RuntimeStatus = RuntimeStatus.INIT,
                  timeout_seconds: int = 120):
        self.bound  = bound
        self.runtime_name = run_time_name
        self.dependencies = dependencies
        self.check_in_func = check_in_func
        self.retry_max_cnt = retry_max_cnt
        self.runtime_status = runtime_status
        self.timeout_seconds = timeout_seconds

    def run(self, **kwargs):
        self._run_with_runtime_info(**kwargs)


    def _run_with_runtime_info(self,**kwargs):
        if self.check_in_func and self.check_in_func(**kwargs):
            for _ in range(self.retry_max_cnt):
                self.bound.run()


    def update_status(self, new_status: RuntimeStatus):
        self.runtime_status = new_status


class BaseExecutor(ABC):

    @abstractmethod
    def run(self,data:BaseData):
        ...

    @abstractmethod
    def run_batch(self,datas: List[BaseData]):
        ...

    @abstractmethod
    def add_module(self,module: BaseModule):
        ...


class ExecutorCallBackHandler(ABC):


    @abstractmethod
    def on_module_done(self,node:RuntimeModule,queue:ThreadSafeQueue,**kwargs):
        ...

    @abstractmethod
    def on_module_error(self,node:RuntimeModule,queue:ThreadSafeQueue,exception:Exception,**kwargs):
        ...

    @abstractmethod
    def on_executor_done(self,):
        ...












