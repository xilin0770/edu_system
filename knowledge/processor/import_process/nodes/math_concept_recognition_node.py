from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, EmbeddingError
from knowledge.processor.import_process.config import get_config
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model
from knowledge.prompts.upload.Classical_Chinese_knowledge import CLASSICAL_CONCEPT_SYSTEM_PROMPT, CLASSICAL_CONCEPT_USER_PROMPT_TEMPLATE

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

class MathConceptRecognitionNode(BaseNode):
    
    name = "math_concept_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        chunks, file_title, config = self._validate_inputs(state)

        # 2. 构建LLM的上下文（目的是为了提取文言文概念）
        classical_chinese_concept_context = self._prepare_classical_chinese_concept_context(chunks, config)

        # 3. 调用LLM模型
        classical_chinese_concept = self._recognition_classical_chinese_concept_by_llm(file_title, classical_chinese_concept_context)

        # 4. 嵌入高中语文文言文概念(
        # 一般的openai或者dashscope的嵌入模型只返回稠密向量
        # bgem3 返回的混合向量：
        # 稠密向量： 提取语义相似性
        # 系数向量： 提取关键词相似性
        # )
        dense, sparse = self._embedding_classical_chinese_concept(classical_chinese_concept)

        # 5. 存储到Milvus数据库
        self._save_to_milvus(file_title, classical_chinese_concept, dense, sparse, config)

        # 6. 回填classical_chinese_concept信息【sate/chunk对象】
        self._fill_classical_chinese_concept(classical_chinese_concept, state, chunks)

        return state
    
    def _fill_classical_chinese_concept(self, classical_chinese_concept: str, state: ImportGraphState, chunks: list[dict]):

        self.log_step("step6", "回填商品名信息")
        for chunk in chunks:
            chunk['classical_chinese_concept'] = classical_chinese_concept  # 方便下游模型能有参考

        state['classical_chinese_concept'] = classical_chinese_concept  # 程序员使用的时候更加方便

    def _embedding_classical_chinese_concept(self, classical_chinese_concept: str) -> tuple[list, dict]:
        self.log_step("step4", "embedding模型嵌入高中语文文言文概念")
        try:
            # 1. 获取嵌入模型
            embedding_model = get_beg_m3_embedding_model()

            # 2. 嵌入classical_chinese_concept
            embedding_result = embedding_model.encode_documents([classical_chinese_concept])

            # 3. 获取稠密和稀疏向量
            dense = embedding_result['dense'][0].tolist()
            start_index = embedding_result['sparse'].indptr[0]
            end_index = embedding_result['sparse'].indptr[1]
            weights = embedding_result['sparse'].data[start_index:end_index].tolist()
            tokenIds = embedding_result['sparse'].indices[start_index:end_index].tolist()
            sparse = dict(zip(tokenIds, weights))
            return dense, sparse

        except Exception as e:
            self.log_step(f"嵌入高中语文文言文概念：{classical_chinese_concept} 失败, 原因是： {str(e)}")
            raise EmbeddingError(f"嵌入高中语文文言文概念：{classical_chinese_concept} 失败, 原因是： {str(e)}", self.name)

    def _validate_inputs(self, state: ImportGraphState):
        self.log_step("step1", "校验输入参数")

        config = get_config()

        # 1. 获取state的file_title以及chunks
        file_title = state.get("file_title")
        chunks = state.get("chunks")
        
        # 2. 判断提取到的参数
        if not file_title:
            raise ValidationError("文件标题为空", self.name)
        
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunks参数为空或不是列表", self.name)
        
        math_concept_chunk_k = config.math_concept_chunk_k
        if not math_concept_chunk_k or math_concept_chunk_k <= 0:
            raise ValidationError("math_concept_chunk_k配置错误", self.name)
        
        self.logger.info(f"检测到文件{file_title}, 对应的切片长度为{len(chunks)}")
        
        # 3. 返回
        return chunks, file_title, config
    
    def _prepare_classical_chinese_concept_context(self, chunks: list, config):

        self.log_step("step2", "构建文言文概念识别上下文")

        spices_result = []

        # 我要从前5块中留下内容的字符数不能超过2000个字符数
        total = 0
        for index, chunk in enumerate(chunks[3:3 + config.math_concept_chunk_k]):
            # 1. 判断chunk的类型
            if not isinstance(chunk, dict):
                continue

            # 构建上下文：[切片-1]标题 + body组成
            # 2. 提取
            body = chunk.get("body")
            spices = f"[切片-{index+1}] content:{body}"
            
            # 3. 计算长度
            total += len(body)
            
            spices_result.append(spices)

            # 判断收集到的长度是否超过最大字符数
            if total > config.math_concept_chunk_size:
                break

        return "\n\n".join(spices_result)[:config.math_concept_chunk_size]
    
    def _recognition_classical_chinese_concept_by_llm(self, file_title: str, classical_chinese_concept_context: str):
        self.log_step("step3", "调用LLM模型识别文言文概念")

        # 1. 获取llm的客户端
        llm_client = get_llm_client()
        if llm_client is None:
            self.logger.error("LLM初始化失败,全回退到标题名{file_title}")
            return file_title

        # 2. 构建LLM的提示词
        prompt = CLASSICAL_CONCEPT_USER_PROMPT_TEMPLATE.format(
            file_title=file_title,
            context=classical_chinese_concept_context
        )

        # 3. 调用LLM模型
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=CLASSICAL_CONCEPT_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            # 4. 提取LLM的回复内容
            content = getattr(llm_response, "content", "").strip()    

            # 5. 判断
            if not content or content.upper() == "UNKNOWN":
                self.logger.warning(f"LLM模型识别数学概念失败,安全回退到标题名{file_title}")      
                return file_title  
            self.logger.info(f"LLM模型识别数学概念成功,数学概念为{content}")
            classical_chinese_concept = content
            return classical_chinese_concept                        
        except Exception as e:
            self.logger.error(f"调用LLM模型失败,安全回退到标题名{file_title}") 
            return file_title 

    def _save_to_milvus(self, file_title: str, classical_chinese_concept: str, dense: list, sparse: dict, config):
        self.log_step("step5", "保存到向量数据库中")
        # 1. 参数检验
        if not dense or not sparse:
            self.logger.warning(f"[{classical_chinese_concept}] 向量生成不完整，跳过入库！")
            return
        
        # 2. 操作Milvus
        try:
            # 2.1 获取Milvus
            milvus_client = get_milvus_client()

            if milvus_client is None:
                return
            
            # 2.2 获取集合的名字
            collection_name = config.classical_chinese_concept_collection

            # 2.3 幂等性校验（不存在则创建新的）
            if not milvus_client.has_collection(collection_name = collection_name):
                self._create_classical_chinese_concept_collection(milvus_client, collection_name)

            # 2.4 构建字典结构数据
            data = {
                "file_title": file_title,  # 文件名字
                "classical_chinese_concept": classical_chinese_concept,  # 高中语文文言文概念
                "dense_vector": dense,  # 稠密向量 （list）
                "sparse_vector": sparse,  # 稀疏向量  (dict:{tokenId:weight})
            }

            # 2.5 插入数据到Milvus:{"insert_count":10,"ids":[10001,10002,10003]}
            result = milvus_client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")
        except Exception as e:
            self.logger.error(f"Milvus 数据库保存操作彻底失败: {e}")

    def _create_classical_chinese_concept_collection(self, milvus_client, collection_name: str):
        self.log_step(f"正在创建集合{collection_name}")
        # 1. 创建约束
        schema = milvus_client.create_schema()

        # 1.1 主键字段的约束
        schema.add_field(
            field_name="pk",
            datatype=DataType.VARCHAR,
            is_primary=True, # 设置主键
            auto_id=True, # 自动分配字段值
            max_length=100
        )

        # 1.2 标量字段的约束
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )

        schema.add_field(
            field_name="classical_chinese_concept",
            datatype=DataType.VARCHAR,
            max_length=65535
        )

        schema.add_field(
            field_name="number",
            datatype=DataType.INT32,
            nullable=True
        )

        # 1.3 向量字段的约束
        # 1.3.1 稠密向量字段约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 1.3.2 稀疏向量字段约束
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引
        index_params = milvus_client.prepare_index_params()
        # 2.1 稠密向量索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index", # 应用此对象后生成的索引文件名称。
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        # 2.2 稀疏向量索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index", # 应用此对象后生成的索引文件名称。
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 3. 创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合{collection_name}创建成功")

if __name__ == "__main__":
    import json
    from pathlib import Path

    base_temp_dir = Path(
        r"D:\pycharm\project\shopkeeper_brain\scripts\processed\语文文言文原文_解析\json_file")

    chunk_json_path = r"D:\pycharm\project\shopkeeper_brain\scripts\processed\语文文言文原文_解析\json_file\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 构建state
    state = {
        "file_title": "文言文",
        "chunks": chunks
    }
    
    math_concept_recognition_node = MathConceptRecognitionNode()
    result = math_concept_recognition_node.process(state=state)

    with open(base_temp_dir / "math_concept_recognition.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

