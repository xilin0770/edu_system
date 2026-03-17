import json, time, re, logging
from concurrent.futures import ThreadPoolExecutor,as_completed
import threading
from json import JSONDecodeError
from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass, field
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import MilvusClient, DataType


from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import ImportConfig
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import Neo4jError, MilvusError
from knowledge.prompts.upload.mathematical_knowledge import MATH_KG_SYSTEM_PROMPT
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.neo4j_util import get_neo4j_driver
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.bge_me_embedding_util import get_beg_m3_embedding_model

# ------------------------------------------
# 常量
# ------------------------------------------
MAX_ENTITY_NAME_LENGTH = 15

# ------------------------------------------
# 白名单   
# ------------------------------------------
# 实体标签白名单 Concept\Formula\Method\Condition\Difficulty
ALLOWED_ENTITY_LABELS: Set[str] = {
    "Concept", "Formula", "Method", "Condition",
    "Difficulty"
}
# 关系类型白名单 BELONGS_TO  PREREQUISITE  REQUIRES_CONDITION  SOLVES  DERIVES_FROM  COMMON_ERROR
ALLOWED_RELATION_TYPES: Set[str] = ({
    "BELONGS_TO", "PREREQUISITE", "REQUIRES_CONDITION", "SOLVES",
    "DERIVES_FROM", "COMMON_ERROR", "MENTIONED_IN", "RELATED_TO"
})
DEFAULT_RELATION_TYPES = "RELATED_TO"

# ------------------------------------------
# Neo4J的Cypher语句
# ------------------------------------------
# Chunk标签节点创建
CYPHER_MERGE_CHUNK = """
    MERGE (c:Chunk {id: $chunk_id, math_concept: $math_concept})
"""

# Entity标签节点的创建
CYPHER_MERGE_ENTITY_TEMPLATE = """
    MERGE (n:Entity {{name: $name, math_concept: $math_concept}})
    ON CREATE SET
        n.source_chunk_id = $chunk_id,
        n.description     = $description
    ON MATCH SET
        n.description = CASE
            WHEN $description <> "" THEN $description
            ELSE coalesce(n.description, "")
        END
    SET n:`{label}`
"""
# Entity关联Chunk
CYPHER_LINK_ENTITY_TO_CHUNK = """
    MATCH (n:Entity {name: $name, math_concept: $math_concept})
    MATCH (c:Chunk  {id: $chunk_id, math_concept: $math_concept})
    MERGE (n)-[:MENTIONED_IN]->(c)
"""

# Entity与Entity的关系
CYPHER_MERGE_RELATION_TEMPLATE = """
    MATCH (h:Entity {{name: $head, math_concept: $math_concept}})
    MATCH (t:Entity {{name: $tail, math_concept: $math_concept}})
    MERGE (h)-[:{rel_type}]->(t)
"""


# 清理Neo4J数据
CYPHER_CLEAR_ITEM = """
    MATCH (n {math_concept: $math_concept}) DETACH DELETE n
"""


@dataclass
class ProcessingStats:
    """处理过程统计信息，用于日志和监控。"""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0

    # 类型注解：List[str] 表示这是一个字符串列表
    # 默认值工厂：default_factory=list 使用 list 函数作为工厂来创建一个新的空列表
    # 避免可变默认值问题：每次实例化对象时都会创建一个新的列表实例，而不是共享同一个列表
    # 如果使用 default=[]，所有实例将共享同一个列表对象，导致意外的数据污染
    errors: List[str] = field(default_factory=list) # 

    def summary(self) -> str:
        return (
            f"处理完成: {self.processed_chunks}/{self.total_chunks} 切片成功, "
            f"{self.failed_chunks} 失败, "
            f"共 {self.total_entities} 实体 / {self.total_relations} 关系"
        )


class _MilvusEntityWriter:
    """负责将实体向量化并写入 Milvus，仅供本模块内部使用。"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def insert(self, milvus_client, entities: List[Dict], chunk_id: str, content: str, math_concept: str) -> None:
        """对外唯一入口：将实体写入 Milvus。"""

        # 1. 判断实体是否存在
        if not entities:
            raise ValueError("参数校验失败，实体不存在")

        # 2. 获取去重后的实体名
        # 这里不能使用set()，因为set是无序的，不能保证实体名的顺序
        # entities_names = set({e["name"] for e in entities})
        entities_names = list(dict.fromkeys(e["name"] for e in entities if e.get("name")))
        if not entities_names:
            raise ValueError("参数校验失败，无有效实体名")

        # 3. 获取嵌入模型
        bge_ef_model = get_beg_m3_embedding_model()

        if bge_ef_model is None:
            raise MilvusError("嵌入模型获取失败")

        # 4. 创建集合（不存在则创建）
        try:
            self._ensure_collection(milvus_client, self.collection_name)
        except Exception as e:
            raise MilvusError(f"Milvus 创建集合失败: {e}")

        # 5. 嵌入向量化
        try:
            embedded_result = bge_ef_model.encode_documents(entities_names)
        except Exception as e:
            raise MilvusError(f"实体嵌入失败: {e}")

        # 6. 构建记录
        records = self._build_records(entities_names, embedded_result, chunk_id, content, math_concept)
        if not records:
            raise MilvusError("构建 Milvus 记录为空")

        # 7. 写入 Milvus
        try:
            milvus_client.insert(collection_name=self.collection_name, data=records)
            self.logger.info(f"Milvus 写入 {len(records)} 条实体向量")
        except Exception as e:
            raise MilvusError(f"Milvus 插入数据失败: {e}")

    def _ensure_collection(self, client, collection_name: str) -> None:
        """集合不存在则创建（schema + 索引）。"""

        # 1. 判断集合是否已存在
        if client.has_collection(collection_name):
            return

        # 2. 构建 schema
        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("entity_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("source_chunk_id", DataType.VARCHAR, max_length=65535)
        schema.add_field("context", DataType.VARCHAR, max_length=65535)
        schema.add_field("math_concept", DataType.VARCHAR, max_length=65535)

        # 3. 构建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 4. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    @staticmethod
    def _build_records(
            entities_names: List[str],
            embedded_result: Dict[str, Any],
            chunk_id: str,
            content: str,
            math_concept: str,
    ) -> List[Dict[str, Any]]:
        """组装插入记录。"""

        # 1. 校验嵌入结果
        if not embedded_result:
            raise ValueError("嵌入结果为空")

        # 2. 获取稠密向量和稀疏向量
        dense_vector_list = embedded_result.get("dense")
        sparse_matrix = embedded_result.get("sparse")

        # 3. 校验向量是否存在
        if not dense_vector_list or sparse_matrix is None:
            raise ValueError("参数校验失败，向量不存在")

        # 4. 获取对应块的部分内容作为上下文
        context = content[:200]
        records: List[Dict] = []

        # 5. 遍历每一个实体名，构建记录
        for idx, entity_name in enumerate(entities_names):
            # 5.1 边界检查
            if idx >= len(dense_vector_list):
                break

            # 5.2 获取稠密向量
            dense = dense_vector_list[idx].tolist()

            # 5.3 解构稀疏向量（从 CSR 矩阵中提取当前实体的稀疏向量）
            start = sparse_matrix.indptr[idx]
            end = sparse_matrix.indptr[idx + 1]
            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()
            sparse_dict = dict(zip(indices, data))

            # 5.4 构建单条记录
            record = {
                "entity_name": entity_name,
                "context": context,
                "math_concept": math_concept,
                "source_chunk_id": chunk_id,
                "dense_vector": dense,
                "sparse_vector": sparse_dict,
            }

            records.append(record)

        return records
    

class _Neo4jGraphWriter:
    def __init__(self, database: str = ""):
        self._database = database
        self._logger = logging.getLogger(self.__class__.__name__)

    def clear(self, neo4j_driver, math_concept: str) -> None:
        if not neo4j_driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(neo4j_driver) as session:
                session.execute_write(
                    lambda tx, name: tx.run(CYPHER_CLEAR_ITEM, math_concept=name),
                    math_concept,
                )
            self._logger.info(f"Neo4j 旧数据已清理: {math_concept}")
        except Exception as e:
            raise Neo4jError(f"Neo4j 清理失败: {e}")
        
    def insert(self, driver, entities, relations, chunk_id: str, math_concept: str):
        """
        Neo4J的写入

        Args:
            driver: neo4j的驱动
            entities:  清洗后的实体
            relations: 清洗后的关系链
            chunk_id:  实体对应的chunk_id
            math_concept: 文档对应LLM提取的商品名

        Returns:

        """
        # 1. 判断实体是否存在
        if not entities:
            raise ValueError("实体不能为空")
        
        # 2. 判断驱动
        if not driver:
            raise Neo4jError("Neo4j 驱动获取失败")
        
        try:
            with self._session(driver) as session:
                session.execute_write(
                    self._write_graph_tx, entities, relations, chunk_id, math_concept
                )
                    
        except Exception as e:
            raise Neo4jError(f"Neo4j 写入失败: {e}")
        
    def _write_graph_tx(self, tx, entities, relations, chunk_id: str, math_concept: str):
        """
        Neo4J的写入事务

        Args:
            tx: neo4j的事务
            entities:  清洗后的实体
            relations: 清洗后的关系链
            chunk_id:  实体对应的chunk_id
            math_concept: 文档对应LLM提取的商品名

        Returns:

        """
        # 1. 创建chunk节点
        tx.run(CYPHER_MERGE_CHUNK, chunk_id=chunk_id, math_concept=math_concept)

        # 2. 创建实体节点+ 关联到chunk
        for entity in entities:
            name = entity.get("name")
            raw_label = entity.get("label")
            description = entity.get("description")

            # 动态格式化cypher， 将安全标签注入（TODO）
            cypher_query = CYPHER_MERGE_ENTITY_TEMPLATE.format(label=raw_label)
            tx.run( cypher_query, name=name, description=description,
                    chunk_id=chunk_id, math_concept=math_concept)

            # 关联实体到 Chunk
            tx.run( CYPHER_LINK_ENTITY_TO_CHUNK,
                    name=name, chunk_id=chunk_id, math_concept=math_concept)
            
        # 3. 创建实体之间的关系
        for rel in relations:
            head = rel.get("head")
            tail = rel.get("tail")
            rel_type = rel.get("type")

            cypher = CYPHER_MERGE_RELATION_TEMPLATE.format(rel_type=rel_type)
            tx.run(cypher, head=head, tail=tail, math_concept=math_concept)

    # 创建数据库会话：为给定的驱动程序创建一个新的数据库会话
    # 参数设置：使用 dataclass=self._database 选项配置会话
    # 返回会话对象：返回创建的会话，用于后续数据库操作
    def _session(self, driver):
        return driver.session(database = self._database)

class KnowLedgeGraphNode(BaseNode):
    name = "knowledge_graph_node"

    def __init__(self, config: Optional[ImportConfig] = None):
        super().__init__(config)
        self._milvus_writer = _MilvusEntityWriter(self.config.entity_name_collection)
        self._neo4j_writer = _Neo4jGraphWriter(self.config.neo4j_database)

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 参数校验
        validated_chunks, math_concept = self._validate_get_inputs(state)

        # 2. 构建统计初始信息
        stats = ProcessingStats(total_chunks=len(validated_chunks))

        # 3. 获取
        # 3.1 获取milvus客户端
        milvus_client = get_milvus_client()
        neo4j_driver = get_neo4j_driver()

        # 4. 删除已经存在的数据（3.1 删除milvus中存储实体名字的记录（delete:math_concept）：幂等性保证 3.2 删除neo4j的整个库下的所有节点以及关系）
        self._clean_exist_double_data(milvus_client, neo4j_driver, math_concept)

        # 5. 批量处理（串行版本） 
        # self._process_all_chunks_v1(stats, validated_chunks, milvus_client, neo4j_driver)
        # 5. 批量处理（多线程版本）
        self._process_chunks_concurrently(stats, validated_chunks, milvus_client, neo4j_driver)
        # 6. 简单的日志观察
        self.logger.info(stats.summary())


    def _clean_exist_double_data(   self, milvus_client: MilvusClient, neo4j_driver,
                                    math_concept: str):
        """
        删除milvus以及neo4j的对应文档的记录
        Args:
            milvus_client:
            neo4j_driver:
            math_concept:

        Returns:

        """
        # 3.1 删除milvus中的math_concept=math_concept的数据
        """导入前清理该 math_concept 下的所有旧数据(Milvus)。"""
        # 1. 清理 Milvus
        if not milvus_client:
            raise MilvusError("Milvus 客户端获取失败")

        collection_name = self.config.entity_name_collection
        try:
            if milvus_client.has_collection(collection_name):
                milvus_client.delete(
                    collection_name=collection_name,
                    filter=f'math_concept == "{math_concept}"',
                )
                self.logger.info(f"Milvus 旧数据已清理: math_concept={math_concept}")
        except Exception as e:
            raise MilvusError(f"Milvus 清理失败: {e}")

        # 3.2 删除neo4j中math_concept=math_concept的数据（实体、关系）

    def _process_all_chunks_v1( self, stats: ProcessingStats,
                                validated_chunks: List[Dict[str, Any]],
                                milvus_client: MilvusClient,
                                neo4j_driver):
        """
        循环处理每一个chunk
        Args:
            validated_chunks:
            milvus_client:
            neo4j_driver:

        Returns:

        """

        # 1. 遍历所有的chunk
        for i, chunk in enumerate(validated_chunks):

            if not isinstance(chunk, dict):
                continue

            # 1.1 获取chunk的信息
            chunk_id = chunk.get('chunk_id')
            math_concept = chunk.get('math_concept')
            content = chunk.get('content')

            # 2. 处理单个chunk
            try:

                entities_count, relations_count = self._process_single_chunk(   chunk_id,
                                                                                math_concept,
                                                                                content,
                                                                                milvus_client,
                                                                                neo4j_driver)
                stats.processed_chunks += 1
                stats.total_entities += entities_count
                stats.total_relations += relations_count
                self.logger.info(f"成功处理完 {chunk_id} / {len(validated_chunks)}")
            except Exception as e:
                stats.failed_chunks += 1
                stats.errors.append(str(e))
                self.logger.error(f"处理失败 {chunk_id} / {len(validated_chunks)}")

    def _process_single_chunk(  self, chunk_id: str,
                                math_concept: str,
                                content: str,
                                milvus_client: MilvusClient,
                                neo4j_driver) -> Tuple[int, int]:

        llm_start = time.time()
        # thread_name = threading.current_thread().name #  获取线程名
        # 1. 调用模型提取chunk的实体、关系
        llm_response = self._extract_graph_with_retry(content)
        llm_cost = time.time() - llm_start

        # 2. 解析并且清洗数据
        graph_result = self._parse_and_clean(llm_response)

        # 2.1 获取解析后的实体
        final_entities = graph_result.get('entities')
        # 2.2 获取解析后的关系
        final_relations = graph_result.get('relations')

        # 3. 写入
        # 3.1 将清洗后的实体名字（可能是多个）存储到milvus
        milvus_start = time.time()
        self._milvus_writer.insert(milvus_client, final_entities, chunk_id, content, math_concept)
        milvus_cost = time.time() - milvus_start

        # 3.2 将清洗后的实体以及关系类型都存储到neo4j
        neo4j_start = time.time()
        self._neo4j_writer.insert(neo4j_driver, final_entities, final_relations, chunk_id, math_concept)
        neo4j_cost = time.time() - neo4j_start

        total_cost = time.time() - llm_start

        # 4. 统计单块处理的时间信息
        self.logger.info(
            # f"[{thread_name}] chunk={chunk_id} | "
            f"实体={len(final_entities)} 关系={len(final_relations)} | "
            f"LLM={llm_cost:.2f}s Milvus={milvus_cost:.2f}s Neo4j={neo4j_cost:.2f}s | "
            f"总计={total_cost:.2f}s"
        )

        return len(final_entities), len(final_relations)

    def _extract_graph_with_retry(self, content: str) -> str:

        # 1. 获取LLM客户端
        llm_client = get_llm_client()
        if llm_client is None:
            raise ValueError(f"LLM客户端初始化失败")

        MAX_COUNT = 3
        last_error = None

        # 2.循环重试3次
        # TODO :将失败的异常原因给到模型
        for attempt in range(1, MAX_COUNT + 1):
            try:
                # 2.1 调用模型
                llm_response = llm_client.invoke([
                    SystemMessage(content=MATH_KG_SYSTEM_PROMPT),
                    HumanMessage(content=f"切片信息\n\n{content}")
                ])
                # 2.2 获取内容
                result = getattr(llm_response, 'content', '').strip()

                # 2.3 有内容
                if result:
                    return result
            except Exception as e:
                last_error = e

                # 2.4 控制重试间隔
                if attempt < MAX_COUNT:
                    # 睡一会：间隔[固定间隔/指数退避]
                    delay = 0.5 * (2 ** (attempt - 1))
                    self.logger.warning(f"开始第{attempt}次重试，间隔：{delay:1.f}s")
                    time.sleep(delay)
        self.logger.error(f"已经进行了{MAX_COUNT}次重试，都失败原因：{str(last_error)}")

        # 3. 最终兜底
        return ""

    def _parse_and_clean(self, llm_response: str) -> Dict[str, Any]:
        """
        1.解析llm返回结果的json代码片段的围栏
        2.反序列化
        3.获取实体信息以及关系信息
        4.分别在清洗实体以及关系
        5. 清洗之后对应的实体和关系返回
        Args:
            llm_response: 模型的输出
        Returns:
            {
                "entities" :[{比较干净的实体名字:标签},{比较干净的实体名字:标签}]
                “relations” :[{比较干净的关系: “head”:"","tail":"","type":""},{比较干净的关系: “head”:"","tail":"","type":""}]
            }
        """

        # 1. 判断
        if not llm_response:
            raise ValueError(f"LLM提取chunk的图谱信息不存在")

        # 2. 清洗json代码块的围栏
        # 2.1 前面的7个非法字符踢掉```json
        # 2.2 后面的3个非法的字符踢掉```
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # 3. 反序列化
        try:
            # cleaned 是一个 JSON 格式的字符串 json.loads() 将其解析为 Python 字典对象
            parsed_llm_response: Dict[str, Any] = json.loads(cleaned)
        except  JSONDecodeError as e:
            raise JSONDecodeError(f"反序列化失败 :{str(e)}")

        # 4. 获取信息
        # 4.1 获取实体信息
        entities = parsed_llm_response.get('entities', [])

        # 4.2 获取关系信息
        relations = parsed_llm_response.get('relations', [])

        # 5. 清洗实体
        cleaned_entities = self._clean_entities(entities)

        # 6. 获取清洗后的实体名
        cleaned_unique_entity_names = {entity.get('name') for entity in cleaned_entities}

        # 7. 清洗关系
        cleaned_relations = self._clean_relations(cleaned_unique_entity_names, relations)

        # 8. 构建返回字典
        return {"entities": cleaned_entities, "relations": cleaned_relations}

    def _clean_entities(self, entities: List[Dict[str,Any]]) -> List[Dict[str, Any]]:
        """
        1. 清洗无效实体（实体名没有）
        2. 阶段过长的实体名（实体名太长）
        3. 实体的标签是否在白名单中
        4. 去重(同名同标签的实体智能存在一份)
        5. 返回
        Args:
            entities: LLM中提取的实体信息

        Returns:
            合法干净的实体信息
        """

        unique_seen = set()
        clean_entities_result = []

        # 1. 遍历所有的实体信息
        for entity in entities:

            # 1.1 获取实体名
            entity_name = str(entity.get('name','')).strip()

            # 1.2 校验名是否存在
            if not entity_name:
                continue

            # 1.3 截取实体名
            if len(entity_name) > MAX_ENTITY_NAME_LENGTH:
                entity_name = entity_name[:15]

            # 1.4 获取实体标签
            entity_label = str(entity.get('label', '')).strip()

            # 1.5 判断标签是否在定义的实体标签是否白名单中
            if entity_label not in ALLOWED_ENTITY_LABELS:
                continue

            # 1.6 定义去重key
            unique_key = (entity_name, entity_label)

            # 1.7 判断是否是同一个实体（实体名+标签）
            if unique_key in unique_seen:
                continue
            unique_seen.add(unique_key)

            # 1.8 构建返回数据结构
            clean_entities = {"name": entity_name, "label": entity_label}

            # 1.9 判断实体的描述
            entity_describe = str(entity.get('description', '')).strip()
            if entity_describe:
                clean_entities['description'] = entity_describe

            # 1.10 将清洗后的实体信息存储到列表
            clean_entities_result.append(clean_entities)

        # 2. 返回最终清洗后的实体列表
        return clean_entities_result

    def _clean_relations(self, cleaned_unique_entity_names: Set[str], relations: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        清洗关系:
        1. 清洗关系的头尾节点是否不存在
        2. 截取头尾实体名过长
        3. 校验头尾实体名是否有效（悬空关系处理）
        4. 校验每一个关系的类型是否关系类型的白名单
        5. 返回

        Args:
            cleaned_unique_entity_names: 所有唯一的实体名集合
            relations:  LLM中提取的关系信息

        Returns:
            List[Dict[str,Any]] 合法干净的关系信息

        """

        clean_relations_result = []

        # 1. 遍历所有的关系
        for relation in relations:

            # 1.1 提取头（head）实体名
            head_entity_name = str(relation.get('head','')).strip()

            # 1.2 提取尾 (tail) 实体名
            tail_entity_name = str(relation.get('tail','')).strip()

            # 1.3 判断头尾实体是否有任意一个不存在
            if not head_entity_name or not tail_entity_name:
                continue

            # 1.4 判断头尾实体名是否超过阈值
            if len(head_entity_name) > MAX_ENTITY_NAME_LENGTH:
                head_entity_name = head_entity_name[:MAX_ENTITY_NAME_LENGTH]

            if len(tail_entity_name) > MAX_ENTITY_NAME_LENGTH:
                tail_entity_name = tail_entity_name[:MAX_ENTITY_NAME_LENGTH]

            # 1.5 判断头尾实体名是否有效
            if head_entity_name not in cleaned_unique_entity_names or tail_entity_name not in cleaned_unique_entity_names:
                continue

            # 1.6 获取关系类型
            relation_type = str(relation.get('type','')).strip()

            # 1.7 判断关系类型是否在关系类型的白名单中
            if relation_type not in ALLOWED_RELATION_TYPES:
                # TODO 思路：反哺白名单
                relation_type = DEFAULT_RELATION_TYPES

            # 1.8 构建最终关系链的数据结构
            cleaned_relation = {"head": head_entity_name, "tail": tail_entity_name, "type": relation_type}

            # 1.9 将清洗后最终的关系链放到最终的结果中
            clean_relations_result.append(cleaned_relation)

        return clean_relations_result

    def _validate_get_inputs(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], str]:
        self.log_step("step1", "知识图谱构建参数校验")

        # 1. 获取基础字段
        chunks = state.get("chunks") or []
        global_math_concept = str(state.get("math_concept", "")).strip()

        # 2. 校验整体 chunks 是否存在
        if not chunks:
            raise ValueError("待提取图谱的切块(chunks)不存在，跳过图谱构建。")

        # 3. 逐个校验 Chunk 的有效性
        validated_chunks = []
        for i, chunk in enumerate(chunks):

            # 3.1 chunk 是否是字典
            if not isinstance(chunk, dict):
                self.logger.warning(f"第 {i} 个 chunk 不是字典类型，已抛弃。")
                continue

            # 3.2 处理 chunk_id
            raw_id = chunk.get("chunk_id")
            chunk_id = str(raw_id).strip() if raw_id is not None else f"kg_chunk_temp_{i}"

            # 3.3 获取 content 内容
            content = str(chunk.get("content", "")).strip()
            if not content:
                self.logger.warning(f"Chunk {chunk_id} 缺少 content，已抛弃。")
                continue

            # 3.4 获取 math_concept（chunk 级别优先，全局兜底）
            chunk_item = str(chunk.get("math_concept", "")).strip() or global_math_concept
            if not chunk_item:
                self.logger.warning(f"Chunk {chunk_id} 缺少 math_concept 归属，已抛弃。")
                continue

            # 3.5 更新 chunk 字段
            chunk["chunk_id"] = chunk_id
            chunk["math_concept"] = chunk_item
            chunk["content"] = content

            # 3.6 加入有效列表
            validated_chunks.append(chunk)

        # 4. 校验清洗后是否还有有效数据
        if not validated_chunks:
            raise ValueError(f"经过清洗后，没有任何有效的 chunk（{len(validated_chunks)}）可用于构建图谱。")

        self.logger.info(f"参数校验完成: 原始 {len(chunks)} 块 -> 有效 {len(validated_chunks)} 块。")

        return validated_chunks, global_math_concept
    
    def _process_chunks_concurrently(self, stats:ProcessingStats, validated_chunks:List[Dict[str,Any]], milvus_client:MilvusClient, neo4j_driver):
        """
        多线程版本：
        多线程本质压榨CPU 和提高响应时间没有本质的关系
        Args:
            stats:
            validated_chunks:
            milvus_client:
            neo4j_driver:
        Returns:

        """

        with ThreadPoolExecutor(max_workers=4) as pool:
            # 1. 提交所有任务
            future_to_idx = {}
            for i, chunk in enumerate(validated_chunks):
                content = chunk.get("content")
                chunk_id = str(chunk.get("chunk_id"))
                math_concept = chunk.get("math_concept")

                # 像线程池中提交任务 返回任务对象
                future = pool.submit(
                    self._process_single_chunk,
                    chunk_id,math_concept, content, milvus_client,neo4j_driver
                )
                future_to_idx[future] = (i, chunk_id)

            # 2. 收集结果（按完成顺序）（一定要让执行_process_chunks_concurrently方法的线程等所有任务做完）
            for future in as_completed(future_to_idx):
                idx, chunk_id = future_to_idx[future]
                try:

                    entity_count, relation_count = future.result()   # 任务的结果（_process_single_chunk 返回值）
                    stats.processed_chunks += 1
                    stats.total_entities += entity_count
                    stats.total_relations += relation_count
                except Exception as e:
                    stats.failed_chunks += 1
                    msg = f"切片 {chunk_id} 处理失败: {e}"
                    stats.errors.append(msg)
                    self.logger.error(msg)


def test_kg_extraction():
    """测试：模拟单个切片，跑通 LLM → 解析 → 清洗全流程。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mock_state = {
        "math_concept": "万用表",
        "chunks": [
        {
            "title": "# 6、参数方程的概念",
            "body": "\n在平面直角坐标系中，如果曲线上任意一点的坐标\n\n$\\mathbf{x},\\mathbf{y}$ 都是某个变数 $\\mathbf{t}$ 的函数 $\\left\\{ \\begin{array}{l}\\mathbf{x} = \\mathbf{f}(\\mathbf{t}),\\\\ \\mathbf{y} = \\mathbf{g}(\\mathbf{t}), \\end{array} \\right.$ 并且对于 $\\mathbf{t}$ 的每一个允许值，由这个方程所确定的点 $\\mathbf{M}(\\mathbf{x},\\mathbf{y})$ 都在这条曲线上，那么这个方程就叫做这条曲线的 参数方\n\n程，联系变数 $x, y$ 的变数 $t$ 叫做参变数，简称参数。相对于参数方程而言，直接给出点的坐标间关系的方程叫 做普通方程。\n",
            "file_title": "万用表的使用",
            "parent_title": "# 6、参数方程的概念",
            "content": "# 6、参数方程的概念\n\n在平面直角坐标系中，如果曲线上任意一点的坐标\n\n$\\mathbf{x},\\mathbf{y}$ 都是某个变数 $\\mathbf{t}$ 的函数 $\\left\\{ \\begin{array}{l}\\mathbf{x} = \\mathbf{f}(\\mathbf{t}),\\\\ \\mathbf{y} = \\mathbf{g}(\\mathbf{t}), \\end{array} \\right.$ 并且对于 $\\mathbf{t}$ 的每一个允许值，由这个方程所确定的点 $\\mathbf{M}(\\mathbf{x},\\mathbf{y})$ 都在这条曲线上，那么这个方程就叫做这条曲线的 参数方\n\n程，联系变数 $x, y$ 的变数 $t$ 叫做参变数，简称参数。相对于参数方程而言，直接给 出点的坐标间关系的方程叫做普通方程。\n",
            "math_concept": "集合 - 集合概念与表示\n函数 - 函数概念与性质\n函数 - 基本初等函数 - 指数函数\n函数 - 基本初等函数 - 对数函数\n函数 - 基本初等函数 - 幂函数\n立体几何 - 立体几何初步\n解析几何 - 平面解析几何初步\n算法 - 算法初步\n统计 - 统计\n概率 - 概率\n函数 - 基本初等函数 - 三角函数\n向量 - 平面向量\n三角函数 - 三角恒等变换\n解三角形 - 解三角形\n数列 - 数列\n不等式 - 不等式\n常用逻辑用语 - 常用逻辑用语\n解析几何 - 圆锥曲线与方程\n微积分 - 导数及其应用\n统计 - 统计案例\n推理与证明 - 推理与证明\n复数 - 数系 的扩充与复数\n算法 - 框图\n立体几何 - 空间向量与立体几何\n计数原理 - 计数原理\n概率 - 随机变量及其分布列\n数学史 - 数学史选讲\n信息安全与密码 - 信息安全与密码\n几何 - 球面上的几何\n代数 - 对称与群\n几何 - 欧拉公式与闭曲面分类\n几何 - 三等分角与数域扩充\n几何 -  几何证明选讲\n代数 - 矩阵与变换\n数列 - 数列与差分\n解析几何 - 坐标系与参数方程\n不等式 - 不等式选讲\n数论 - 初等数论初步\n优选法与试验设计 - 优选法与试验设计初步\n统筹法与图论 - 统筹法与图论初步\n风险与决策 - 风险与决策\n开关电路与布尔代数 - 开关电路与布尔代数",
            "dense_vector": [
                0.0253753662109375,
                -0.005298614501953125,
                -0.0239410400390625,
                -0.034393310546875,
                0.03240966796875,
                -0.0274810791015625,
                0.01519012451171875,
                -0.0025272369384765625,
                0.0394287109375,
                -0.0049896240234375,
                -0.0298004150390625,
                0.01157379150390625,
                0.0178680419921875,
                -0.005702972412109375,
                0.0194549560546875,
                0.017425537109375,
                0.01276397705078125,
                0.02056884765625,
                0.031646728515625,
                -0.002231597900390625,
                -0.042724609375,
                0.01241302490234375,
                -0.0004181861877441406,
                -0.01027679443359375,
                -0.0292205810546875,
                -0.01398468017578125,
                0.0192718505859375,
                0.0205841064453125,
                -0.0297088623046875,
                0.0271453857421875,
                0.004154205322265625,
                0.0035266876220703125,
                0.03887939453125,
                -0.036834716796875,
                -0.06610107421875,
                0.0142669677734375,
                -0.031707763671875,
                -0.021331787109375,
                -0.030487060546875,
                0.040374755859375,
                -0.0230712890625,
                -0.030059814453125,
                0.040740966796875,
                -0.039642333984375,
                0.0029697418212890625,
                -0.035919189453125,
                0.00601959228515625,
                -0.0712890625,
                -0.0011625289916992188,
                -0.04132080078125,
                -0.03741455078125,
                -0.0152130126953125,
                0.051910400390625,
                -0.09368896484375,
                -0.046722412109375,
                0.0013704299926757812,
                -0.0621337890625,
                -0.06060791015625,
                -0.06597900390625,
                -0.01406097412109375,
                -0.046783447265625,
                0.039886474609375,
                -0.07421875,
                0.00684356689453125,
                0.03277587890625,
                0.0017728805541992188,
                -0.0008053779602050781,
                -0.00704193115234375,
                -0.0019483566284179688,
                -0.01812744140625,
                -0.0030345916748046875,
                0.019012451171875,
                -0.0173187255859375,
                -0.0148773193359375,
                -0.0215301513671875,
                0.0222930908203125,
                -0.0175323486328125,
                0.0013494491577148438,
                -0.029541015625,
                0.01427459716796875,
                -0.0202484130859375,
                -0.015869140625,
                -0.0023345947265625,
                -0.05694580078125,
                0.033538818359375,
                0.02520751953125,
                0.0281524658203125,
                0.00145721435546875,
                -0.004909515380859375,
                -0.0170440673828125,
                0.032135009765625,
                0.01239776611328125,
                0.01474761962890625,
                -0.01100921630859375,
                -0.06573486328125,
                -0.0367431640625,
                -0.0218048095703125,
                0.0152587890625,
                -0.0011644363403320312,
                0.0013637542724609375,
                0.00543212890625,
                0.052001953125,
                0.02569580078125,
                -0.007080078125,
                -0.0015115737915039062,
                0.03118896484375,
                0.01190185546875,
                -0.00347137451171875,
                0.007747650146484375,
                -0.0240631103515625,
                0.032440185546875,
                0.01389312744140625,
                0.0233154296875,
                0.009185791015625,
                -0.042236328125,
                -0.004840850830078125,
                0.00746917724609375,
                -0.00930023193359375,
                0.0180511474609375,
                0.01361846923828125,
                0.043304443359375,
                0.0196533203125,
                0.054931640625,
                -0.0184326171875,
                0.001678466796875,
                -0.00572967529296875,
                0.0249176025390625,
                0.00572967529296875,
                -0.055145263671875,
                0.001865386962890625,
                -0.031982421875,
                -0.014739990234375,
                -0.0302734375,
                -0.00848388671875,
                -0.036956787109375,
                -0.011260986328125,
                -0.005428314208984375,
                0.0196380615234375,
                -0.0202178955078125,
                -0.07086181640625,
                0.022186279296875,
                -0.00737762451171875,
                -0.035736083984375,
                -0.0256195068359375,
                0.0250244140625,
                -0.0264739990234375,
                0.00913238525390625,
                0.03448486328125,
                -0.021240234375,
                -0.017059326171875,
                0.01103973388671875,
                0.04217529296875,
                -0.0272369384765625,
                0.050689697265625,
                -0.04705810546875,
                -0.014190673828125,
                -0.04840087890625,
                0.0128021240234375,
                0.0174407958984375,
                -0.0032196044921875,
                -0.0103607177734375,
                0.0184173583984375,
                0.004009246826171875,
                -0.0104522705078125,
                0.043853759765625,
                0.01322174072265625,
                -0.0213623046875,
                0.052093505859375,
                0.01039886474609375,
                -0.052215576171875,
                -0.013397216796875,
                -0.0158538818359375,
                0.0184783935546875,
                -0.007411956787109375,
                0.0235748291015625,
                0.0501708984375,
                0.09136962890625,
                0.018798828125,
                0.00359344482421875,
                -0.0087127685546875,
                -0.0131683349609375,
                0.02215576171875,
                0.0196075439453125,
                -0.01012420654296875,
                0.019256591796875,
                0.021575927734375,
                -0.0193634033203125,
                -0.0234375,
                -0.020477294921875,
                0.0140838623046875,
                0.006008148193359375,
                -0.04400634765625,
                0.03363037109375,
                0.024871826171875,
                0.07672119140625,
                -0.021697998046875,
                -0.00762939453125,
                -0.003467559814453125,
                -0.0123748779296875,
                -0.0209197998046875,
                -0.032012939453125,
                0.024200439453125,
                0.01342010498046875,
                0.002544403076171875,
                -0.046966552734375,
                0.0034732818603515625,
                -0.0882568359375,
                -0.07110595703125,
                0.024566650390625,
                -0.060699462890625,
                -0.0042724609375,
                0.01500701904296875,
                0.0227508544921875,
                0.0175628662109375,
                0.0247955322265625,
                0.03558349609375,
                -0.0343017578125,
                0.007152557373046875,
                0.033843994140625,
                -0.0097808837890625,
                0.0002486705780029297,
                0.01099395751953125,
                -0.0293121337890625,
                0.030548095703125,
                0.03717041015625,
                -0.03759765625,
                -0.025299072265625,
                0.01812744140625,
                0.039031982421875,
                -0.006908416748046875,
                -0.022003173828125,
                0.017486572265625,
                0.026885986328125,
                -0.027191162109375,
                -0.019561767578125,
                -0.050048828125,
                -0.04510498046875,
                -0.0044708251953125,
                0.020782470703125,
                0.0022182464599609375,
                -0.0390625,
                -0.0218048095703125,
                0.028045654296875,
                -0.032379150390625,
                -0.0303192138671875,
                0.020477294921875,
                0.03485107421875,
                0.0206451416015625,
                -0.01462554931640625,
                -0.038970947265625,
                0.01125335693359375,
                0.031463623046875,
                -0.03399658203125,
                -0.0084991455078125,
                0.023681640625,
                0.01276397705078125,
                0.039215087890625,
                -0.0008649826049804688,
                -0.027069091796875,
                -0.0215301513671875,
                -0.0284881591796875,
                0.037506103515625,
                0.0301361083984375,
                0.0153045654296875,
                0.031402587890625,
                0.0004706382751464844,
                -0.0236053466796875,
                -0.01397705078125,
                0.0075531005859375,
                0.019866943359375,
                -0.021453857421875,
                -0.01678466796875,
                0.033447265625,
                -0.01776123046875,
                -0.0247802734375,
                0.03265380859375,
                -0.00423431396484375,
                -0.003002166748046875,
                0.068603515625,
                -0.019683837890625,
                0.01043701171875,
                -0.017547607421875,
                0.0416259765625,
                0.006214141845703125,
                0.0643310546875,
                -0.01873779296875,
                -0.06182861328125,
                -0.033203125,
                0.0093841552734375,
                -0.031982421875,
                -0.02764892578125,
                0.00020384788513183594,
                0.07220458984375,
                -0.01264190673828125,
                -0.017791748046875,
                -0.031402587890625,
                0.006500244140625,
                -0.1522216796875,
                0.03265380859375,
                0.0051116943359375,
                0.0172271728515625,
                -0.00021731853485107422,
                0.03253173828125,
                -0.0716552734375,
                -0.00699615478515625,
                0.025787353515625,
                0.054718017578125,
                -0.02392578125,
                -0.03741455078125,
                -0.002712249755859375,
                -0.055633544921875,
                -0.0296783447265625,
                0.0125885009765625,
                -0.01360321044921875,
                0.034637451171875,
                0.0460205078125,
                -0.025665283203125,
                0.02398681640625,
                -0.032379150390625,
                0.044769287109375,
                0.025360107421875,
                -0.0197906494140625,
                0.027374267578125,
                0.01522064208984375,
                0.061309814453125,
                0.00018715858459472656,
                0.0008721351623535156,
                -0.00815582275390625,
                0.01113128662109375,
                0.006710052490234375,
                0.00569915771484375,
                0.016448974609375,
                0.0146636962890625,
                -0.016876220703125,
                0.0022869110107421875,
                0.00455474853515625,
                0.0086822509765625,
                0.00954437255859375,
                0.046295166015625,
                -0.019775390625,
                0.039642333984375,
                -0.02069091796875,
                0.013885498046875,
                0.019989013671875,
                -0.01213836669921875,
                -0.0267333984375,
                0.0264739990234375,
                -0.0294647216796875,
                0.01313018798828125,
                -0.0286102294921875,
                -0.00439453125,
                -0.05206298828125,
                0.030853271484375,
                -0.0083770751953125,
                0.02313232421875,
                -0.037628173828125,
                0.039520263671875,
                -0.0161895751953125,
                -0.038848876953125,
                0.0176239013671875,
                -0.03350830078125,
                -0.01201629638671875,
                -0.03125,
                0.030242919921875,
                0.0168609619140625,
                -0.0016689300537109375,
                -0.036865234375,
                0.0302734375,
                -0.0126495361328125,
                0.01727294921875,
                -0.003597259521484375,
                -0.022003173828125,
                0.007965087890625,
                0.005939483642578125,
                0.036102294921875,
                -0.01497650146484375,
                -0.0589599609375,
                0.01605224609375,
                0.0015668869018554688,
                -0.00390625,
                0.00724029541015625,
                -0.0465087890625,
                0.0210418701171875,
                0.0259246826171875,
                -0.043731689453125,
                0.00191497802734375,
                0.2210693359375,
                0.06671142578125,
                -0.01302337646484375,
                -0.0196075439453125,
                0.0136871337890625,
                0.0074920654296875,
                0.0021877288818359375,
                -0.00848388671875,
                -0.030242919921875,
                -0.00772857666015625,
                -0.01544189453125,
                0.00644683837890625,
                0.0272979736328125,
                -0.0285491943359375,
                0.002536773681640625,
                0.07147216796875,
                -0.0567626953125,
                0.0159912109375,
                0.03802490234375,
                0.01552581787109375,
                0.02862548828125,
                0.05120849609375,
                0.049041748046875,
                -0.0175018310546875,
                -0.0697021484375,
                0.00965118408203125,
                -0.0121002197265625,
                0.0301361083984375,
                -0.0167388916015625,
                -0.01280975341796875,
                -0.0235748291015625,
                0.0281982421875,
                0.036041259765625,
                -0.08355712890625,
                -0.01480865478515625,
                -0.05120849609375,
                0.0212860107421875,
                -0.0218353271484375,
                0.0338134765625,
                0.01052093505859375,
                0.06707763671875,
                0.0248260498046875,
                -0.03302001953125,
                -0.050567626953125,
                -0.028045654296875,
                -0.0421142578125,
                0.0037593841552734375,
                0.0196990966796875,
                0.05474853515625,
                -0.053558349609375,
                0.0156402587890625,
                -0.0340576171875,
                -0.04864501953125,
                -0.019622802734375,
                -0.050933837890625,
                0.01360321044921875,
                0.01461029052734375,
                0.059112548828125,
                0.01910400390625,
                0.001373291015625,
                0.06744384765625,
                0.0129241943359375,
                0.02154541015625,
                0.006439208984375,
                -0.012847900390625,
                -0.0209197998046875,
                0.0411376953125,
                -0.029571533203125,
                0.06439208984375,
                0.0309295654296875,
                0.026458740234375,
                0.0212860107421875,
                0.0142364501953125,
                0.0421142578125,
                0.035247802734375,
                -0.03533935546875,
                0.0018548965454101562,
                0.0233306884765625,
                -0.0116729736328125,
                -0.048980712890625,
                -0.043487548828125,
                -0.0231475830078125,
                -0.00010073184967041016,
                -0.010040283203125,
                0.02130126953125,
                -0.0015249252319335938,
                -0.0156402587890625,
                0.042083740234375,
                -0.0017910003662109375,
                -0.01094818115234375,
                -0.0141448974609375,
                -0.0301513671875,
                -0.0015277862548828125,
                -0.0005064010620117188,
                -0.004352569580078125,
                0.031524658203125,
                -0.0261993408203125,
                -0.0247802734375,
                -0.0645751953125,
                0.035888671875,
                0.050811767578125,
                0.0150909423828125,
                -0.0110626220703125,
                0.007350921630859375,
                0.0210418701171875,
                -0.09539794921875,
                0.0030117034912109375,
                0.01287078857421875,
                0.00405120849609375,
                -0.024749755859375,
                -0.07745361328125,
                0.0199737548828125,
                0.006420135498046875,
                -0.004856109619140625,
                0.05340576171875,
                0.00843048095703125,
                -0.001056671142578125,
                0.0117034912109375,
                0.01171875,
                0.0248565673828125,
                -0.0017786026000976562,
                0.0157623291015625,
                -0.0108795166015625,
                0.01922607421875,
                -0.01125335693359375,
                0.0002701282501220703,
                -0.019805908203125,
                -0.0090179443359375,
                0.0240936279296875,
                0.0126800537109375,
                0.0323486328125,
                -0.031646728515625,
                -0.0174560546875,
                0.0002543926239013672,
                0.0174560546875,
                0.0193023681640625,
                0.01406097412109375,
                0.0299072265625,
                0.026336669921875,
                0.0155487060546875,
                -0.035675048828125,
                -0.022857666015625,
                -0.0008072853088378906,
                0.027801513671875,
                0.0233917236328125,
                0.0596923828125,
                0.025665283203125,
                -0.005229949951171875,
                -0.048980712890625,
                0.0267791748046875,
                0.0089569091796875,
                -0.005382537841796875,
                0.0253448486328125,
                -0.0193939208984375,
                -0.0097808837890625,
                -0.041656494140625,
                -0.0034847259521484375,
                -0.02020263671875,
                0.007293701171875,
                -0.00494384765625,
                -0.004940032958984375,
                -0.00543975830078125,
                0.004974365234375,
                0.0338134765625,
                0.01534271240234375,
                0.007503509521484375,
                -0.003509521484375,
                -0.03851318359375,
                0.0013666152954101562,
                -0.02911376953125,
                -0.00977325439453125,
                -0.0462646484375,
                -0.0341796875,
                0.003223419189453125,
                0.0260009765625,
                0.018890380859375,
                -0.0303802490234375,
                0.0149078369140625,
                -0.036041259765625,
                0.002353668212890625,
                0.03955078125,
                0.0115509033203125,
                0.054779052734375,
                0.016998291015625,
                -0.019195556640625,
                0.0236968994140625,
                -0.0025463104248046875,
                0.004169464111328125,
                0.00013267993927001953,
                0.01275634765625,
                -0.055816650390625,
                0.129638671875,
                -0.06256103515625,
                -0.043731689453125,
                0.02032470703125,
                0.056793212890625,
                0.040313720703125,
                -0.00043272972106933594,
                -0.004802703857421875,
                -0.072021484375,
                0.046295166015625,
                0.023468017578125,
                -0.025543212890625,
                -0.0016069412231445312,
                0.0157623291015625,
                0.0152740478515625,
                0.0428466796875,
                0.0036945343017578125,
                -0.039031982421875,
                -0.004589080810546875,
                0.01580810546875,
                -0.02166748046875,
                -0.054443359375,
                0.020782470703125,
                -0.0158233642578125,
                -0.03802490234375,
                0.0185089111328125,
                0.00841522216796875,
                -0.0487060546875,
                -0.01428985595703125,
                0.0209197998046875,
                -0.0175018310546875,
                -0.031463623046875,
                -0.037628173828125,
                -0.0238494873046875,
                -0.00997161865234375,
                -0.01255035400390625,
                -0.0167694091796875,
                -0.019134521484375,
                0.0019283294677734375,
                -0.00814056396484375,
                0.007808685302734375,
                0.00627899169921875,
                -0.00823974609375,
                -0.00841522216796875,
                -0.0167083740234375,
                0.04595947265625,
                0.01898193359375,
                0.0125885009765625,
                -0.03350830078125,
                0.00406646728515625,
                0.031890869140625,
                0.00872802734375,
                0.02001953125,
                -0.040985107421875,
                -0.0034637451171875,
                -0.05206298828125,
                -0.01201629638671875,
                -0.0238189697265625,
                0.03857421875,
                -0.007495880126953125,
                0.0012006759643554688,
                0.0299835205078125,
                0.006557464599609375,
                0.0175628662109375,
                0.005390167236328125,
                0.011993408203125,
                0.034698486328125,
                -0.0087890625,
                0.00821685791015625,
                0.0041351318359375,
                0.01666259765625,
                0.0221710205078125,
                0.00283050537109375,
                -0.009552001953125,
                -0.053741455078125,
                0.0091705322265625,
                -0.08197021484375,
                -0.043853759765625,
                -0.03485107421875,
                -0.0172119140625,
                -0.005870819091796875,
                -0.0523681640625,
                0.00693511962890625,
                -0.035186767578125,
                -0.028289794921875,
                -0.0364990234375,
                -0.0382080078125,
                -0.052581787109375,
                -0.009429931640625,
                -0.04833984375,
                5.751848220825195e-05,
                -0.005645751953125,
                -0.006977081298828125,
                -0.01251220703125,
                0.03582763671875,
                -0.02496337890625,
                0.03887939453125,
                0.045440673828125,
                -0.0330810546875,
                0.02392578125,
                0.006534576416015625,
                -0.0099334716796875,
                0.048736572265625,
                0.028076171875,
                0.01885986328125,
                -0.018890380859375,
                0.052947998046875,
                -0.0220947265625,
                0.030120849609375,
                -0.0078887939453125,
                0.01464080810546875,
                -0.01385498046875,
                0.004543304443359375,
                -0.03076171875,
                -1.430511474609375e-05,
                -0.00518035888671875,
                -0.049468994140625,
                0.01031494140625,
                0.0262603759765625,
                -0.02545166015625,
                -0.0343017578125,
                -0.0020694732666015625,
                0.051727294921875,
                0.0174102783203125,
                0.0020427703857421875,
                0.0606689453125,
                -0.048553466796875,
                0.03533935546875,
                0.00397491455078125,
                -0.048004150390625,
                -0.0184173583984375,
                -0.0207061767578125,
                -0.0245819091796875,
                0.00885772705078125,
                -0.021820068359375,
                -0.0086212158203125,
                -0.030731201171875,
                -0.01018524169921875,
                -0.0311431884765625,
                -0.019134521484375,
                -0.002506256103515625,
                -0.0166778564453125,
                -0.01180267333984375,
                -0.0177001953125,
                -0.003253936767578125,
                -0.0197906494140625,
                -0.027099609375,
                0.0008034706115722656,
                -0.0186767578125,
                0.03424072265625,
                -0.07586669921875,
                0.055023193359375,
                -0.052001953125,
                0.062744140625,
                0.01456451416015625,
                -0.006084442138671875,
                -0.0211334228515625,
                -0.0239105224609375,
                0.0258026123046875,
                -0.010772705078125,
                0.0016021728515625,
                0.0078887939453125,
                -0.054718017578125,
                -0.028106689453125,
                -0.03399658203125,
                0.0308990478515625,
                -0.0126190185546875,
                0.0645751953125,
                0.0306549072265625,
                0.01097869873046875,
                0.03729248046875,
                0.07904052734375,
                -0.039306640625,
                -0.07098388671875,
                0.03857421875,
                0.01380157470703125,
                -0.0298309326171875,
                0.033660888671875,
                -0.0186004638671875,
                -0.017578125,
                0.036956787109375,
                0.030426025390625,
                -0.00234222412109375,
                -0.00951385498046875,
                0.031646728515625,
                -0.040496826171875,
                -0.0230560302734375,
                0.00927734375,
                -0.0190277099609375,
                -0.038360595703125,
                -0.04852294921875,
                -0.044097900390625,
                -0.0229949951171875,
                -0.000659942626953125,
                0.0005064010620117188,
                -0.0127716064453125,
                -0.005077362060546875,
                0.0254058837890625,
                0.035400390625,
                -0.0007004737854003906,
                0.0129241943359375,
                -0.00019097328186035156,
                0.019256591796875,
                -0.13720703125,
                0.0232391357421875,
                -0.001750946044921875,
                -0.0025882720947265625,
                -0.031951904296875,
                0.0003592967987060547,
                -0.037933349609375,
                0.0133819580078125,
                0.0276031494140625,
                -0.047882080078125,
                0.00843048095703125,
                0.0227508544921875,
                0.035186767578125,
                -0.007328033447265625,
                0.01520538330078125,
                0.004184722900390625,
                0.003757476806640625,
                -0.0215301513671875,
                -0.025970458984375,
                0.0268096923828125,
                0.0163421630859375,
                -0.0288848876953125,
                0.08685302734375,
                0.0190277099609375,
                0.034515380859375,
                -0.053802490234375,
                0.0294036865234375,
                -0.032806396484375,
                -0.0016908645629882812,
                -0.04229736328125,
                0.017364501953125,
                -0.0230865478515625,
                0.029327392578125,
                0.0257720947265625,
                0.018463134765625,
                -0.0006189346313476562,
                -0.00711822509765625,
                -0.0017681121826171875,
                -0.017547607421875,
                0.01512908935546875,
                0.0333251953125,
                0.00527191162109375,
                -0.048065185546875,
                -0.007579803466796875,
                -0.0296478271484375,
                0.036590576171875,
                0.01491546630859375,
                -0.0242767333984375,
                -0.0086212158203125,
                0.034759521484375,
                0.0186767578125,
                0.0185546875,
                0.0287933349609375,
                0.08251953125,
                0.0119171142578125,
                -0.01096343994140625,
                -0.016693115234375,
                0.01519012451171875,
                -0.0294647216796875,
                0.0224151611328125,
                0.0112152099609375,
                0.01259613037109375,
                -0.0164794921875,
                -0.036773681640625,
                -0.01377105712890625,
                -0.016021728515625,
                0.045654296875,
                0.0159454345703125,
                -0.01068115234375,
                -0.033355712890625,
                -9.447336196899414e-05,
                0.01910400390625,
                -0.01149749755859375,
                0.0244293212890625,
                0.02069091796875,
                -0.0477294921875,
                0.03717041015625,
                0.0006051063537597656,
                -0.0309295654296875,
                0.0180511474609375,
                -0.006908416748046875,
                0.0172576904296875,
                -0.049652099609375,
                0.047027587890625,
                0.0241851806640625,
                -0.0300140380859375,
                -0.0123748779296875,
                -0.0169830322265625,
                -0.035736083984375,
                -0.0024738311767578125,
                -0.06304931640625,
                -0.0182037353515625,
                0.017578125,
                0.0007190704345703125,
                5.167722702026367e-05,
                0.0240325927734375,
                -0.0218963623046875,
                0.02020263671875,
                0.0143280029296875,
                -0.027801513671875,
                0.016143798828125,
                -0.0198822021484375,
                0.017669677734375,
                0.025543212890625,
                -0.024078369140625,
                0.005489349365234375,
                0.012725830078125,
                0.0013179779052734375,
                0.0223236083984375,
                0.052398681640625,
                -0.01392364501953125,
                -0.0254058837890625,
                -0.0625,
                0.004970550537109375,
                -0.015625,
                0.0113983154296875,
                -0.020904541015625,
                0.036529541015625,
                -0.00853729248046875,
                0.033905029296875,
                0.0146942138671875,
                0.0560302734375,
                0.023895263671875,
                -0.03204345703125,
                0.01220703125,
                0.0260009765625,
                0.0149993896484375,
                0.0016040802001953125,
                0.023590087890625,
                0.0193634033203125,
                0.00540924072265625,
                -0.0188140869140625,
                0.02191162109375,
                0.032379150390625,
                0.0172576904296875,
                0.020111083984375,
                -0.04229736328125,
                0.056793212890625,
                -0.051361083984375,
                -0.042938232421875,
                0.0095672607421875,
                -0.069091796875,
                0.0293426513671875,
                -0.002788543701171875,
                0.0650634765625,
                -0.00904083251953125,
                -0.0227203369140625,
                0.0251007080078125,
                -0.021240234375,
                -0.02099609375,
                0.0794677734375,
                -0.047393798828125,
                -0.027618408203125,
                0.0204620361328125,
                -0.01739501953125,
                0.0193634033203125,
                -0.06170654296875,
                -0.01558685302734375,
                -0.0123748779296875,
                0.01418304443359375,
                0.0328369140625,
                -0.01947021484375,
                -0.0117034912109375,
                0.004001617431640625,
                -0.01666259765625,
                0.0186004638671875,
                -0.0192718505859375,
                0.02435302734375,
                0.005687713623046875,
                -0.00879669189453125,
                0.0340576171875,
                0.03399658203125,
                -0.0276336669921875,
                -0.02972412109375,
                0.0151214599609375,
                -0.05743408203125,
                0.0228118896484375,
                -0.00879669189453125,
                0.03729248046875,
                -0.0170440673828125,
                0.01146697998046875,
                0.0185699462890625,
                0.048980712890625,
                -0.01209259033203125,
                0.040283203125,
                0.0105438232421875,
                -0.009063720703125,
                0.01340484619140625,
                0.035400390625,
                0.0147247314453125,
                -0.0306396484375,
                -0.0002982616424560547,
                0.0035400390625,
                -0.0282745361328125,
                0.0023097991943359375,
                -0.0115814208984375,
                -0.00902557373046875,
                0.00824737548828125,
                -0.01313018798828125,
                0.033905029296875,
                -0.032989501953125,
                0.035858154296875,
                -0.023590087890625,
                -0.043182373046875,
                -0.006000518798828125,
                -0.023895263671875,
                -0.01259613037109375,
                0.020721435546875,
                -0.02978515625,
                0.043609619140625,
                0.0182952880859375,
                -0.01617431640625,
                -0.0178680419921875,
                0.083984375,
                0.00762939453125,
                -0.01345062255859375,
                0.00334930419921875,
                0.057403564453125,
                -0.01361846923828125,
                -0.0139007568359375,
                0.035675048828125,
                -0.0110931396484375,
                -0.040924072265625,
                -0.02178955078125,
                -0.04638671875,
                0.0063629150390625,
                -0.0201873779296875,
                -0.01546478271484375,
                0.00440216064453125,
                -0.036285400390625,
                0.017547607421875,
                -0.039520263671875,
                -0.008544921875,
                0.0256195068359375,
                0.03619384765625,
                0.042449951171875,
                -0.004199981689453125,
                0.0248260498046875,
                0.0251617431640625,
                -0.01291656494140625,
                0.055877685546875,
                -0.0212554931640625,
                0.024993896484375,
                -0.006282806396484375
            ],
            "sparse_vector": {
                "6": 0.0193634033203125,
                "18": 0.0921630859375,
                "37": 0.044281005859375,
                "43": 0.020965576171875,
                "53": 0.081787109375,
                "113": 0.062347412109375,
                "141": 0.0755615234375,
                "177": 0.05072021484375,
                "305": 0.13330078125,
                "420": 0.0831298828125,
                "425": 0.0631103515625,
                "468": 0.03375244140625,
                "514": 0.0214080810546875,
                "594": 0.0657958984375,
                "2213": 0.09747314453125,
                "2391": 0.12493896484375,
                "2480": 0.07818603515625,
                "2698": 0.2025146484375,
                "3352": 0.0977783203125,
                "3436": 0.0654296875,
                "3611": 0.0089569091796875,
                "3650": 0.0137481689453125,
                "4723": 0.142822265625,
                "6071": 0.1458740234375,
                "6376": 0.01337432861328125,
                "6618": 0.052032470703125,
                "6820": 0.00315093994140625,
                "9005": 0.11492919921875,
                "9557": 0.1484375,
                "10050": 0.0297393798828125,
                "10502": 0.09173583984375,
                "10956": 0.1376953125,
                "11072": 0.2333984375,
                "11158": 0.1417236328125,
                "11233": 0.1370849609375,
                "12514": 0.1563720703125,
                "14685": 0.10223388671875,
                "14841": 0.20361328125,
                "18318": 0.1060791015625,
                "19305": 0.00522613525390625,
                "21446": 0.15966796875,
                "23204": 0.16162109375,
                "24854": 0.0081787109375,
                "25322": 0.1700439453125,
                "27644": 0.190185546875,
                "31741": 0.000614166259765625,
                "32538": 0.09295654296875,
                "41872": 0.0120086669921875,
                "54969": 0.06304931640625,
                "56243": 0.09356689453125,
                "86435": 0.1640625,
                "86731": 0.1474609375,
                "109010": 0.106689453125,
                "109227": 0.254638671875,
                "117350": 0.0292510986328125,
                "125458": 0.0129241943359375,
                "141870": 0.03533935546875,
                "143964": 0.1729736328125,
                "149106": 0.042724609375,
                "150373": 0.08953857421875,
                "150598": 0.165771484375,
                "159619": 0.0872802734375,
                "203593": 0.318115234375,
                "227330": 0.2054443359375
            },
            "chunk_id": 464971605952120727
        }
    ],
}

    knowledge_graph_node = KnowLedgeGraphNode()

    knowledge_graph_node.process(mock_state)


if __name__ == "__main__":
    test_kg_extraction()
