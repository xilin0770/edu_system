from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.bge_me_embedding_util import get_beg_m3_embedding_model, generate_hybrid_embedding
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.processor.query_process.exceptions import StateFieldEooror

import json


class VectorSearchNode(BaseNode):

    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        validated_query, validated_entity_names = self._validate_query_inputs(state)

        # 2. 获取嵌入模型以及milvus客户端
        embedding_model = get_beg_m3_embedding_model()
        milvus_client = get_milvus_client()

        if embedding_model is None or milvus_client is None:
            return state

        # 3. 对问题向量化
        embedding_result = generate_hybrid_embedding(embedding_documents=[validated_query], embedding_model=embedding_model)
        if not embedding_result:
            return state
        
        # 4, 构建过滤表达式
        entity_name_filter_expr = self._entity_name_filter(validated_entity_names)

        # 5. 创建混合搜索请求
        hybrid_requests = create_hybrid_search_requests(
            dense_vector=embedding_result["dense"][0],
            sparse_vector=embedding_result["sparse"][0],
            expr = entity_name_filter_expr,
            limit = 5
        )

        # 5. 执行混合搜索请求
        reps =  execute_hybrid_search_query(
            milvus_client = milvus_client,
            collection_name= "kb_graph_entity_names",
            search_requests = hybrid_requests,
            ranker_weights=(0.5, 0.5),
            norm_score=True,
            output_fields=["source_chunk_id", "context"]
            )
        
        if not reps or not reps[0]:
            return state

        # 6. 更新state的embedding_chunks
        state["embedding_chunks"] = reps[0]
        return state


    def _validate_query_inputs(self, state: QueryGraphState) -> tuple[str, list[str]]:
        """校验查询输入参数。

        Args:
            state: 查询状态对象。

        Returns:
            校验后的查询字符串和商品名称列表。

        Raises:
            StateFieldEooror: 状态字段缺失或无效。
        """
        # 1. 获取state的rewritten_query字段
        rewritten_query = state.get("rewritten_query", "")

        # 2. 获取state中的entity_names字段
        entity_names = state.get("entity_names", "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldEooror(field_name="rewritten_query", node_name=self.name, expected_type = str)
        
        if not entity_names or not isinstance(entity_names, list):
            raise StateFieldEooror(field_name="entity_names", node_name=self.name, expected_type = list)
        
        # 4. 返回
        return rewritten_query, entity_names
    
    def _entity_name_filter(self, entity_names: list[str]) -> str:
        """构建商品名称过滤表达式。

        Args:
            entity_names: 商品名称列表。

        Returns:
            商品名称过滤表达式。
        """
        # conditions = [f"entity_name in ['{name}']" for name in entity_names]
        conditions = [f"entity_name like '%{name}%'" for name in entity_names]
        return "(" + " OR ".join(conditions) + ")"

    
if __name__ == "__main__":
    # 测试代码
    state = {
        'original_query': '18年的Ⅰ卷中“鲁芝字世英，扶风郿人也，世有名德，为西州豪族。”中的“也”有什么作用？',
        'rewritten_query': '请问“也”字在文言文句子“鲁芝字世英，扶风郿人也，世有名德，为西州豪族。”中有什么作用？',
        'entity_names': ['也']
    }
    node = VectorSearchNode()
    new_state = node.process(state)
    for r in new_state.get("embedding_chunks"):
        print(json.dumps(r, ensure_ascii=False, indent=2))

"""
{
  "pk": 465024973573397209,
  "distance": 0.6291012763977051,
  "entity": {
    "source_chunk_id": "465024973573395373",
    "context": "1.魏文侯以为将：省略句，省略了为将的人，应为“魏文侯以之为将”。\n\n2.乃以为西河守：省略句，应为“乃以之为西河守”。\n\n3.与士卒最下者同衣食：定语后置，“最下者”修饰“士卒”，应为“与最下士卒同衣食”。\n\n4. 此魏国之宝也 / 此乃吾所以居子之上也：判断句，“此……也”为判断句标志。"
  }
}
{
  "pk": 465024973573396251,
  "distance": 0.35837340354919434,
  "entity": {
    "source_chunk_id": "465024973573395311",
    "context": "请简要概述孔子三次回答的内容，并说明此则短文反映了孔子怎样的思想。\n\n答案：孔子三次回答的内容分别是：\n\n（1）孔子回答子路，听到了什么不要直接那样去做，要向父亲、兄长请教。\n\n（2）孔子回答冉有，听到了就要去做。\n\n（3）孔子向公西华 解释，为什么同样的问题，不同的人问会有两种不同的回答。\n\n子路性格勇猛所以要让他谨慎，冉有性格怯懦所以要鼓励他前进。此则短文 反映的是孔子因材施教的思想。\n\n解析：本"
  }
}
{
  "pk": 465024973573396239,
  "distance": 0.3583691716194153,
  "entity": {
    "source_chunk_id": "465024973573395306",
    "context": "子路问：“闻斯行诸？”子曰：“有父兄在，如之何其闻斯行之？”\n\n冉有问：“闻斯行诸？”子曰：“闻斯行之。”\n\n公西华 曰：“由也问闻斯行诸，子曰：‘有父兄在。’求也问闻斯行诸，子曰：‘闻斯行之。赤也惑，敢问。”\n\n子曰：“求也退，故进之；由也兼人， 故退之。”"
  }
}
{
  "pk": 465024973573396257,
  "distance": 0.3581934869289398,
  "entity": {
    "source_chunk_id": "465024973573395312",
    "context": "仁是孔子思想的主要体现，不同的弟子在向老师请教仁的内涵的时候，孔子也是给予不\n\n同的回答。比如说《论语·颜渊篇》中颜回问仁的时候，孔子的回答是“克己复礼，天下归仁焉，为仁由己，而由人乎哉”。弟子仲弓问仁的时候，孔子的回答是“己所不欲，勿施于人”。司马牛问仁的时候，孔子的回答是“仁者其言也忍”。樊迟来向孔子请教仁的时候，孔子的回答是“爱人”。在《论语》中，不同的弟子向孔子询问同样的问题，孔子针对每一"
  }
}
{
  "pk": 465024973573396210,
  "distance": 0.2628641426563263,
  "entity": {
    "source_chunk_id": "465024973573395310",
    "context": "1.闻斯行诸：代词，这些、这样。\n\n2.闻斯行诸：兼词，“之乎”。“之”为代词，“乎”为疑问语气词。\n\n3.赤也惑：句中 语气词，表停顿。\n\n4.求也退：句中语气词，表停顿。"
  }
}
"""
