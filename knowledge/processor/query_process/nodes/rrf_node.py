from typing import List, Dict, Any, Tuple

from knowledge.processor.query_process.state import QueryProcessState
from knowledge.processor.query_process.base import BaseNode


class RrfNode(BaseNode):
    """
    rrf: 多路结果的融合,基于文档的排名把多路都命中的文档,未来计算出来的得分就更高,相对应的顺序就越靠前
    """

    name = "rrf_node"

    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k


    def process(self, state: QueryProcessState) ->QueryProcessState:
        # 1. 各路搜索结果(排除网络搜索这一路, 在rerank节点中实现) ---> 本质是因为网络搜索的结果没有chunk_id
        # 1.1 获取向量检索路的结果
        vector_search_chunks = state.get("embedding_chunks") or []
        # 1.2 获取hyde向量检索路的结果
        hyde_search_chunks = state.get("hyde_embedding_chunks") or []
        # 1.3 获取kg图谱检索的结果
        kg_search_chunks = state.get("kg_chunks") or []

        # 2. 为不同路的搜索结果设置不同的权重
        vector_search = {
            "vector_search_result" : (self._normalize_result(vector_search_chunks), 1.0),
            "hyde_search_result" : (self._normalize_result(hyde_search_chunks), 1.0),
            "kg_search_result" : (self._normalize_result(kg_search_chunks), 0.7)
        }
        # 2.1 构建rrf的inputs
        rrf_inputs = list(vector_search.values()) 

        # 3. 利用RRF的计算公式去获取到所有路查询到的所有chunk对应的score
        # 当rrf_k更大时 最后的结果偏爱于多路有相同chunk_id的文档，
        # 而当rrf_k比较小时，最后的结果更偏爱于在某一条路靠前的
        merge_result: List[Dict[str, Any]] = self._rrf_merge(rrf_inputs, self._top_k, self._rrf_k)

        # 4. 最终返回
        return merge_result

    def _normalize_result(self, rrf_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        统一处理各路搜索到的结果
        Args:
            rrf_input: 各路不同数据结构的检索结果

        Returns:
            统一处理后的标准数据结构的检索结果
        """

        diff_path_result = []
        # 1. 遍历各路搜索结果
        if not rrf_input:
            return []
        
        # 2. 遍历该路的所有结果
        for doc in rrf_input:
            # 2.1 判断
            if isinstance(doc, dict):
                continue
            # 2.2 获取entity
            entity = doc.get("entity")
            if not entity:
                continue

            diff_path_result.append(entity)
        return diff_path_result
    
    def _rrf_merge(self, rrf_inputs: List[Dict[str, Any]], top_k: int, rrf_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        利用RRF的计算公式去获取到所有路查询到的所有chunk对应的score
        Args:
            rrf_inputs: 各路搜索结果的输入
            top_k: 最终返回的top_k个结果
            rrf_k: RRF的平滑参数一般60

        rrf: sum(weight_i / (rrf_k + rank_i))

        Returns:
            所有路查询到的所有chunk对应的score
        """
        chunk_score = {} # 存放所有chunk的score
        chunk_data = {}  # 存放所有chunk
        for rrf_input,rrf_weight in rrf_inputs:
            # i 为这一路的排名
            for i, doc in enumerate(rrf_input, 1):
                chunk_id = doc.get("chunk_id")
                if not chunk_id:
                    continue

                chunk_score[chunk_id] = chunk_score.get(chunk_id, float(0)) + rrf_weight / (rrf_k + i)

                chunk_data.setdefault(chunk_id, doc)

        sorted_results = sorted(
            [(chunk_data[cid], score) for cid, score in chunk_score.items()],
            key=lambda x: x[1], reverse=True
        )

        return sorted_results[:top_k] if top_k else sorted_results
    
if __name__ == "__main__":


    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟三路检索结果
    # chunk_1 命中 3 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4, chunk_5 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
        "kg_chunks": [
            {"id": None, "distance": 2.0, "entity": {"chunk_id": "chunk_5", "content": "知识图谱结果#1"}},
            {"id": None, "distance": 1.0, "entity": {"chunk_id": "chunk_1", "content": "知识图谱结果#2"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print(f"  kg_chunks: {len(mock_state['kg_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")
