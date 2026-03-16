"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# ABC (Abstract Base Class): 抽象基类，用于定义抽象类
# 继承自 ABC 的类不能被直接实例化
# 用于定义接口和规范子类必须实现的方法

# abstractmethod: 抽象方法装饰器
# 用于标记必须在子类中实现的抽象方法
# 如果子类没有实现被 @abstractmethod 标记的方法，实例化时会抛出错误

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ImportProcessError
T = TypeVar("T")  # 泛型状态类型 任意变量类型（可以是字典、列表、字符串等）


class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        # 设置节点配置，优先使用传入的配置，否则使用全局配置
        self.config = config or get_config()
        # 创建日志记录器，命名格式为 "import.{节点名称}"
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """

        self.logger.info(f"--- {self.name} 开始 ---")


        try:
            result = self.process(state)
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except ImportProcessError:
            # 已经是自定义异常，直接抛出
            raise
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod # 这里的装饰器代表每个子类必须实现这个方法，不实现会报错
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑

        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        print("开始执行process")
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志

        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)



# 配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置导入流程日志

    Args:
        level: 日志级别(如 logging.INFO, logging.DEBUG, logging.ERROR)
    %(asctime)s：日志产生的时间（具体的显示格式由下面的 datefmt 决定）
    %(name)s：记录器（Logger）的名字。在复杂的系统中，通常用当前模块名（如 query.rerank）来追踪是哪个文件打印的。
    %(levelname)s：日志的级别名称（比如 INFO、ERROR）
    %(message)s：真正在代码里想打印的具体内容（比如 logger.info("合并文档: 4 篇") 里面的那句话）。
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # 日志格式
        datefmt='%Y-%m-%d %H:%M:%S'
    )


if __name__ == "__main__":
    basenode = BaseNode()
