import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import json

from langgraph.graph import StateGraph, END

from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.nodes.math_concept_recognition_node import MathConceptRecognitionNode
from knowledge.processor.import_process.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.import_milvus_node  import ImportMilvusNode
from knowledge.processor.import_process.nodes.kg_graph_node import KnowLedgeGraphNode
from knowledge.processor.import_process.state import create_default_state
from knowledge.processor.import_process.base import setup_logging

from langgraph.graph import StateGraph

def import_router(state: ImportGraphState) -> str:
    """
    导入路由函数
    """
    if state.get("is_md_read_enabled"):
        return "md_img_node"
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    return END


def create_import_graph() -> StateGraph:
    """
    定义导入业务的graph状态拓扑图(langgraph构建流水线)整个流水线各个节点要读取或者写入的节点
    Returns:

    """
    # print("------------开始定义状态图-----------------")
    # 1. 定义状态图
    graph_pineline = StateGraph(ImportGraphState)

    # print("------------状态图定义完成-----------------")
    # 2. 定义节点（入口、结束节点、自己要添加的）
    ## 2.1 定义入口节点
    graph_pineline.set_entry_point("entry_node")

    ## 2.2 添加剩下的节点
    # 先实例化之后添加
    # print("------------开始实例化节点-----------------")
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "document_split_node":DocumentSplitNode(),
        "math_concept_recognition":MathConceptRecognitionNode(),
        "bge_embedding_node":BgeEmbeddingChunksNode(),
        "import_milvus_node":ImportMilvusNode(),
        "kg_node":KnowLedgeGraphNode()
    }
    # print("------------节点实例化完成-----------------")

    for key, value in nodes.items():
        print(f"添加节点:{key}")
        graph_pineline.add_node(key, value)

    # 3. 定义边（顺序边、条件边）

    # 条件边
    # source : 路由开始节点
    # path : 路由函数
    # path_map : 路由函数返回的路径映射表
    graph_pineline.add_conditional_edges(
        source="entry_node",
        path= import_router,
        path_map={
            "md_img_node": "md_img_node", # 前面时路由函数返回的，后面的是节点名字
            "pdf_to_md_node": "pdf_to_md_node",
            END: END
        }
    )

    graph_pineline.add_edge("entry_node", "pdf_to_md_node")
    graph_pineline.add_edge("pdf_to_md_node", "document_split_node") # END节点不需要自己实例化
    graph_pineline.add_edge("document_split_node","math_concept_recognition")
    graph_pineline.add_edge("math_concept_recognition","bge_embedding_node")
    graph_pineline.add_edge("bge_embedding_node","import_milvus_node")
    graph_pineline.add_edge("import_milvus_node","kg_node")
    graph_pineline.add_edge("md_img_node", END)
    # print("------------边定义完成-----------------")

    # 4.编译（编排）
    # print("------------开始编译图-----------------")
    return graph_pineline.compile()
    

graph_app = create_import_graph()



# 测试使用
def run_import_graph(import_file_path: str, file_dir: str):
    # 1. 构建state
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    init_state = create_default_state(**state)  # 发生了解包
    print(f"初始状态:{init_state}")

    # 2. 调用stream(用流式获取每一个节点的处理情况：event事件[节点名字 节点处理后的状态])
    final_state = None
    a = graph_app.stream(init_state)
    # print(f"图的流:{a}")
    for event in a:
        # print(f"当前事件:{event}")

        for node_name, state in event.items():
            print(f"运行节点的:{node_name},state:{state}")
            final_state = state

    return final_state


if __name__ == '__main__':
    setup_logging()

    import_file_path = r"D:\pycharm\project\shopkeeper_brain\scripts\Original document\高中数学知识点归纳.pdf"
    file_dir = r"D:\pycharm\project\shopkeeper_brain\scripts\processed"
    # 1. 测试编排流程
    final_state = run_import_graph(import_file_path=import_file_path, file_dir=file_dir)
    print(json.dumps(f'最终状态:{final_state}', indent=2, ensure_ascii=False))

    # 2.打印图结构（ASCII 可视化）# 1. 单独安装：pip install grandalf 2.(单独安装还出错)  【pydantic：定义数据模型 】pip uninstall gradio  3. 单独安装 pip install grandalf 解决冲突
    print("-" * 50)
    print("图结构:")
    graph_app.get_graph().print_ascii()