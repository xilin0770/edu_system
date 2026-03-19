from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.bge_me_embedding_util import get_beg_m3_embedding_model, generate_hybrid_embedding
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.processor.query_process.exceptions import StateFieldEooror
from knowledge.utils.llm_client import get_llm_client
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE

from langchain_core.messages import SystemMessage, HumanMessage
import json

class HyDeSearchNode(BaseNode):
    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        # 1. 参数校验
        validated_query, validated_item_names = self._validate_query_inputs(state)

        # 2. 生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validated_item_names)

        # 3. 获取嵌入模型以及milvus客户端
        embedding_model = get_beg_m3_embedding_model()
        milvus_client = get_milvus_client()

        if  not embedding_model or not milvus_client:
            return state

        # 4. 对假设性文档的嵌入
        embedding_document = f"{validated_query}\n {hy_document}"
        embedding_result = generate_hybrid_embedding(embedding_documents=[embedding_document], embedding_model=embedding_model)
        if not embedding_result:
            return state
        
        # 5. 获取item_name的过滤表达式
        item_name_filter_expr = self._item_name_filter(validated_item_names)
        
        # 6. 创建混合搜索请求
        hybird_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result["dense"][0],
            sparse_vector=embedding_result["sparse"][0],
            expr=item_name_filter_expr
        )

        # 7. 执行混合搜索请求
        reps = execute_hybrid_search_query(
            milvus_client = milvus_client,
            collection_name= "test_chunk",
            search_requests = hybird_search_requests,
            ranker_weights=(0.5, 0.5),
            norm_score=True,
            output_fields=["chunk_id", "content", "item_name"]
        )

        if not reps or not reps[0]:
            return state
        
        # 8. 更新state的hyde_embedding_chunks
        state["hyde_embedding_chunks"] = reps[0]

        # 9. 返回更新后的state
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

        # 2. 获取state中的item_names字段
        item_names = state.get("item_names", "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldEooror(field_name="rewritten_query", node_name=self.name, expected_type = str)
        
        if not item_names or not isinstance(item_names, list):
            raise StateFieldEooror(field_name="item_names", node_name=self.name, expected_type = list)
        
        # 4. 返回
        return rewritten_query, item_names

    def _generate_hy_document(self, query: str, validated_item_names: list[str]) -> str:

        # 1. 获取llm的客户端
        llm_client = get_llm_client()

        if llm_client is None:
            return ""
        
        # 3. 获取系统提示词
        user_prompt = USER_HYDE_PROMPT_TEMPLATE.format(item_hint = validated_item_names, rewritten_query = query)
        system_prompt = f"你是一位{validated_item_names}的技术文档领域的装甲，主要擅长编写技术文档、操作手册、文档规格说明"
        try:
            # 4. 获取AIMessage
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 5. 获取内容
            llm_response_content = getattr(llm_response, "content", "").strip()

            # 6. 判断是否存在
            if not llm_response_content:
                return ""
            
            return llm_response_content
        except Exception as e:
            self.logger.error(f"调用LLM调用失败: {e}")
            return ""
        
    def _item_name_filter(self, item_names: list[str]) -> str:
        """构建商品名称过滤表达式。

        Args:
            item_names: 商品名称列表。

        Returns:
            商品名称过滤表达式。
        """
        # conditions = [f"item_name in ['{name}']" for name in item_names]
        conditions = [f"item_name like '%{name}%'" for name in item_names]
        return "(" + " OR ".join(conditions) + ")"
    

if __name__ == "__main__":
    # 测试代码
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["数字万用表 RS-12"]
    }
    node = HyDeSearchNode()
    new_state = node.process(state)

    for r in new_state.get("hyde_embedding_chunks", []):
        print(json.dumps(r, ensure_ascii=False, indent=2))


