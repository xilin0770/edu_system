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
from knowledge.prompts.upload.Classical_Chinese_knowledge import CLASSICAL_KNOWLEDGE_GRAPH_SYSTEM_PROMPT
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.neo4j_util import get_neo4j_driver
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model

# ------------------------------------------
# 常量
# ------------------------------------------
MAX_ENTITY_NAME_LENGTH = 15

# ------------------------------------------
# 白名单   
# ------------------------------------------
# 实体标签白名单 
ALLOWED_ENTITY_LABELS: Set[str] = {
    "Character", "Grammar_Rule", "Cultural_Term", "Contextual_Sense",
    "Pitfall"
}
# 关系类型白名单 
ALLOWED_RELATION_TYPES: Set[str] = ({
    "POLYSEMY", "INFLECTION", "SYNTACTIC_STRUCTURE", "ANCIENT_MODERN_CONTRAST",
    "CULTURAL_CONTEXT", "LOGICAL_INFERENCE", "MENTIONED_IN", "RELATED_TO"
})
DEFAULT_RELATION_TYPES = "RELATED_TO"

# ------------------------------------------
# Neo4J的Cypher语句
# ------------------------------------------
# Chunk标签节点创建
CYPHER_MERGE_CHUNK = """
    MERGE (c:Chunk {title: $title, id: $chunk_id, chinese_concept: $chinese_concept})
"""

# Entity标签节点的创建
CYPHER_MERGE_ENTITY_TEMPLATE = """
    MERGE (n:Entity {{name: $name, chinese_concept: $chinese_concept}})
    ON CREATE SET
        n.source_chunk_id = $chunk_id,
        n.title           = $title,
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
    MATCH (n:Entity {name: $name, chinese_concept: $chinese_concept, title: $title})
    MATCH (c:Chunk  {title: $title, id: $chunk_id, chinese_concept: $chinese_concept})
    MERGE (n)-[:MENTIONED_IN]->(c)
"""

# Entity与Entity的关系
CYPHER_MERGE_RELATION_TEMPLATE = """
    MATCH (h:Entity {{name: $head, chinese_concept: $chinese_concept}})
    MATCH (t:Entity {{name: $tail, chinese_concept: $chinese_concept}})
    MERGE (h)-[:{rel_type}]->(t)
"""


# 清理Neo4J数据
CYPHER_CLEAR_ITEM = """
    MATCH (n {chinese_concept: $chinese_concept}) DETACH DELETE n
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

    def insert(self, milvus_client, entities: List[Dict], chunk_id: str, body: str, chinese_concept: str) -> None:
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
        records = self._build_records(entities_names, embedded_result, chunk_id, body, chinese_concept)
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
        schema.add_field("chinese_concept", DataType.VARCHAR, max_length=65535)

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
            body: str,
            chinese_concept: str,
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
        context = body[:200]
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
                "chinese_concept": chinese_concept,
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
        
    def insert(self, driver, entities, relations, chunk_id: str, chinese_concept: str, title: str):
        """
        Neo4J的写入

        Args:
            driver: neo4j的驱动
            entities:  清洗后的实体
            relations: 清洗后的关系链
            chunk_id:  实体对应的chunk_id
            chinese_concept: 文档对应LLM提取的商品名

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
                    self._write_graph_tx, entities, relations, chunk_id, chinese_concept, title
                )
                    
        except Exception as e:
            raise Neo4jError(f"Neo4j 写入失败: {e}")
        
    def _write_graph_tx(self, tx, entities, relations, chunk_id: str, chinese_concept: str, title: str):
        """
        Neo4J的写入事务

        Args:
            tx: neo4j的事务
            entities:  清洗后的实体
            relations: 清洗后的关系链
            chunk_id:  实体对应的chunk_id
            chinese_concept: 文档对应LLM提取的文言文概念

        Returns:

        """
        # 1. 创建chunk节点
        tx.run(CYPHER_MERGE_CHUNK, title=title, chunk_id=chunk_id, chinese_concept=chinese_concept)

        # 2. 创建实体节点+ 关联到chunk
        for entity in entities:
            name = entity.get("name")
            raw_label = entity.get("label")
            description = entity.get("description")

            # 动态格式化cypher， 将安全标签注入（TODO）
            cypher_query = CYPHER_MERGE_ENTITY_TEMPLATE.format(label=raw_label)
            tx.run( cypher_query, name=name, description=description,
                    chunk_id=chunk_id, chinese_concept=chinese_concept, title=title)

            # 关联实体到 Chunk
            tx.run( CYPHER_LINK_ENTITY_TO_CHUNK,
                    name=name, chunk_id=chunk_id, chinese_concept=chinese_concept, title=title)
            
        # 3. 创建实体之间的关系
        for rel in relations:
            head = rel.get("head")
            tail = rel.get("tail")
            rel_type = rel.get("type")

            cypher = CYPHER_MERGE_RELATION_TEMPLATE.format(rel_type=rel_type)
            tx.run(cypher, head=head, tail=tail, chinese_concept=chinese_concept)

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
        validated_chunks = self._validate_get_inputs(state)

        # 2. 构建统计初始信息
        stats = ProcessingStats(total_chunks=len(validated_chunks))

        # 3. 获取
        # 3.1 获取milvus客户端
        milvus_client = get_milvus_client()
        neo4j_driver = get_neo4j_driver()

        # 4. 删除已经存在的数据（3.1 删除milvus中存储实体名字的记录（delete:math_concept）：幂等性保证 3.2 删除neo4j的整个库下的所有节点以及关系）
        self._clean_exist_double_data(milvus_client, neo4j_driver, validated_chunks[0]["classical_chinese_concept"])

        # 5. 批量处理（串行版本） 
        # self._process_all_chunks_v1(stats, validated_chunks, milvus_client, neo4j_driver)
        # 5. 批量处理（多线程版本）
        self._process_chunks_concurrently(stats, validated_chunks, milvus_client, neo4j_driver)
        # 6. 简单的日志观察
        self.logger.info(stats.summary())


    def _clean_exist_double_data(   self, milvus_client: MilvusClient, neo4j_driver,
                                    chinese_concept: str):
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
                    filter=f'chinese_concept == "{chinese_concept}"',
                )
                self.logger.info(f"Milvus 旧数据已清理: chinese_concept={chinese_concept}")
            self.logger.info(f"没有旧数据")
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
            chinese_concept = chunk.get("classical_chinese_concept")
            body = chunk.get('body')
            title = chunk.get('title')

            # 2. 处理单个chunk
            try:

                entities_count, relations_count = self._process_single_chunk(   chunk_id,
                                                                                chinese_concept,
                                                                                body,
                                                                                title,
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
                                chinese_concept: str,
                                body: str,
                                title:str,
                                milvus_client: MilvusClient,
                                neo4j_driver) -> Tuple[int, int]:

        llm_start = time.time()
        # thread_name = threading.current_thread().name #  获取线程名
        # 1. 调用模型提取chunk的实体、关系
        llm_response = self._extract_graph_with_retry(body)
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
        self._milvus_writer.insert(milvus_client, final_entities, chunk_id, body, chinese_concept)
        milvus_cost = time.time() - milvus_start

        # 3.2 将清洗后的实体以及关系类型都存储到neo4j
        neo4j_start = time.time()
        self._neo4j_writer.insert(neo4j_driver, final_entities, final_relations, chunk_id, chinese_concept, title)
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

    def _extract_graph_with_retry(self, body: str) -> str:

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
                    SystemMessage(content=CLASSICAL_KNOWLEDGE_GRAPH_SYSTEM_PROMPT),
                    HumanMessage(content=f"切片信息\n\n{body}")
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
                    self.logger.warning(f"开始第{attempt}次重试，间隔：{delay:.1f}s")
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

            # 3.3 获取 body 内容
            body = str(chunk.get("body", "")).strip()
            if not body:
                self.logger.warning(f"Chunk {chunk_id} 缺少 body，已抛弃。")
                continue


            # 3.5 更新 chunk 字段
            chunk["chunk_id"] = chunk_id
            chunk["body"] = body

            # 3.6 加入有效列表
            validated_chunks.append(chunk)

        # 4. 校验清洗后是否还有有效数据
        if not validated_chunks:
            raise ValueError(f"经过清洗后，没有任何有效的 chunk（{len(validated_chunks)}）可用于构建图谱。")

        self.logger.info(f"参数校验完成: 原始 {len(chunks)} 块 -> 有效 {len(validated_chunks)} 块。")

        return validated_chunks
    
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
                body = chunk.get("body")
                chunk_id = str(chunk.get("chunk_id"))
                chinese_concept = chunk.get("classical_chinese_concept")
                title = chunk.get("title")[2:]

                # 像线程池中提交任务 返回任务对象
                future = pool.submit(
                    self._process_single_chunk,
                    chunk_id,chinese_concept, body, title, milvus_client,neo4j_driver
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
        "chunks": [
        {
            "title": "# 2018 全国Ⅰ卷",
            "body": "\n鲁芝字世英，扶风郿人也，世有名德，为西州豪族。父为郭氾所害，芝襁褓流离，年十七，乃移居雍，耽思坟籍。郡举上计吏，州辟别驾。魏车骑将军郭淮为雍州刺史，深敬重之。举孝廉，除郎中，后拜骑都尉、参军事、行安南太守，迁尚书郎，曹真出督关右，又参大司马军事。真薨，宣帝代焉，乃引芝参骠骑军事，转天水太守。郡邻于蜀，数被侵掠，户口减削，寇盗充斥，芝倾心镇卫，更造城市，数年间旧境悉复。迁广平太守，天水夷夏慕德，老幼赴阙献书，乞留芝，魏明帝许焉。曹爽辅政，引为司马，芝屡有谠言嘉谋，爽弗能纳。及宣帝起兵诛爽，芝率余众犯门斩关，驰出赴爽，劝爽曰：“公居伊周之位，一旦以罪见黜，虽欲牵黄犬，复可得乎！若挟天子保许昌，杖大威以羽檄征四方兵，孰敢不从！舍此而去，欲就东市，岂不痛哉！”爽懦惑不能用遂委身受戮芝坐爽下狱当死而口不讼直志不苟免宣帝嘉之赦而不诛俄而起为并州刺史。诸葛诞以寿春叛，魏帝出征，芝率荆州文武以为先驱。诞平，迁大尚书，掌刑理，武帝践阼，转镇东将军，进爵为侯。帝以芝清忠履正，素无居宅，使军兵为作屋五十间。芝以年及悬车，告老逊位，章表十余上，于是征为光禄大夫，位特进，给吏卒，门施行马。羊祜为车骑将军，乃以位让芝，曰：“光禄大夫鲁芝洁身寡欲，和而不同，服事华发，以礼终始，未蒙此选，臣更越之，何以塞天下之望！”上不从。其为人所重如是。泰始九年卒，年八十四。帝为举哀，谥曰贞，赐茔田百亩。\n\n（节选自《晋书·鲁芝传》）\n",
            "file_title": "文言文",
            "parent_title": "# 2018 全国Ⅰ卷",
            "classical_chinese_concept": "文言实词 - 一词多义 - “除”的多种含义\n文言实词 - 一词多义 - “迁”的多种含义\n文言实词 - 一词多义 - “引”的多种含义\n文言实词 - 古今异义 - “城市”的含义\n文言实词 - 古今异义 - “先驱”的含义\n文言实词 - 词类活用 - 名词作状语 - “襁褓流离”\n文言实词 - 词类活用 - 使动用法 - “更造城市”\n文言句式 - 被动句 - “为...所...”结构\n文言句式 - 被动句 - “见”字表被动\n文言句式 - 省略句 - 主语省略\n文言句式 - 倒装句 - 宾语前置 - “孰敢不从”\n文言句式 - 判断句 - “...也”表判断\n文化常识 - 官职制度 - 举孝廉、除郎中、拜骑都尉、迁尚书郎\n文化常识 - 官职制度 - 别驾、太守、刺史\n文化常识 - 称谓 - 字、谥号\n文化常识 - 历史典故 - 伊周、牵黄犬、东市\n内容理解 - 人物形象分析 - 鲁芝的性格品质\n内容理解 - 人物形象分析 - 鲁芝的仕途经历\n内容理解 - 文意概括 - 主要事件梳理",
            "dense_vector": [
                -0.0309295654296875,
                0.052398681640625,
                -0.047821044921875,
                -0.0013437271118164062,
                0.0053863525390625,
                -0.003787994384765625,
                0.0027179718017578125,
                0.033905029296875,
                0.0269012451171875,
                0.0118255615234375,
                -0.047637939453125,
                0.0012865066528320312,
                -0.049041748046875,
                0.0013065338134765625,
                0.03533935546875,
                0.008819580078125,
                0.018646240234375,
                -0.007007598876953125,
                -0.0153045654296875,
                -0.03741455078125,
                0.01139068603515625,
                -0.016265869140625,
                0.013885498046875,
                0.053619384765625,
                0.03277587890625,
                0.017608642578125,
                -0.02239990234375,
                -0.026641845703125,
                -0.021026611328125,
                0.0268707275390625,
                0.01030731201171875,
                -0.0016536712646484375,
                -0.0179901123046875,
                -0.00846099853515625,
                0.00843048095703125,
                0.025909423828125,
                0.0037689208984375,
                -0.0703125,
                -0.016998291015625,
                -0.0130462646484375,
                0.03631591796875,
                0.038604736328125,
                0.05242919921875,
                0.003665924072265625,
                -0.0106964111328125,
                -0.03973388671875,
                0.002628326416015625,
                -0.01334381103515625,
                -0.0006380081176757812,
                0.0179595947265625,
                0.002201080322265625,
                -0.0164642333984375,
                -0.003711700439453125,
                0.02606201171875,
                -0.0217742919921875,
                -0.0182647705078125,
                0.05523681640625,
                0.0268096923828125,
                0.0171051025390625,
                -0.01190185546875,
                0.00909423828125,
                -0.0309906005859375,
                -0.033203125,
                -0.007457733154296875,
                0.034912109375,
                0.083984375,
                0.006732940673828125,
                0.00797271728515625,
                -0.0161590576171875,
                -0.0038890838623046875,
                -0.01120758056640625,
                -0.01444244384765625,
                0.0265350341796875,
                -0.0005784034729003906,
                -0.0506591796875,
                0.0284576416015625,
                0.038055419921875,
                0.0149078369140625,
                -0.00322723388671875,
                0.035125732421875,
                0.032257080078125,
                -0.0007886886596679688,
                0.005985260009765625,
                0.0210723876953125,
                -0.07275390625,
                -0.005550384521484375,
                0.02325439453125,
                0.029052734375,
                -0.0120849609375,
                0.004825592041015625,
                -0.0357666015625,
                -0.0278778076171875,
                0.013519287109375,
                -0.03973388671875,
                0.00897979736328125,
                -0.022796630859375,
                -0.056549072265625,
                -0.01062774658203125,
                0.084228515625,
                -0.0229949951171875,
                -0.0158843994140625,
                0.05059814453125,
                0.01055908203125,
                -0.01268768310546875,
                0.047393798828125,
                -0.027252197265625,
                -0.0012950897216796875,
                0.0148162841796875,
                -0.0445556640625,
                0.00804901123046875,
                0.0136566162109375,
                0.036651611328125,
                -0.015228271484375,
                -0.033599853515625,
                -0.0262908935546875,
                -0.0175323486328125,
                0.0188140869140625,
                -0.020782470703125,
                -0.013275146484375,
                0.01030731201171875,
                0.028106689453125,
                -0.0168304443359375,
                -0.0038356781005859375,
                -0.00839996337890625,
                0.01418304443359375,
                0.006023406982421875,
                0.01317596435546875,
                0.0157470703125,
                0.050140380859375,
                0.0177459716796875,
                -0.0086822509765625,
                0.03271484375,
                0.01384735107421875,
                -0.003162384033203125,
                -0.0543212890625,
                0.0276947021484375,
                0.057373046875,
                0.035675048828125,
                0.042572021484375,
                -0.01873779296875,
                0.007343292236328125,
                0.01314544677734375,
                0.0225830078125,
                -0.005340576171875,
                0.0496826171875,
                -0.026458740234375,
                0.0271759033203125,
                0.005634307861328125,
                -0.013214111328125,
                0.00015783309936523438,
                -0.01364898681640625,
                -0.03265380859375,
                0.0174560546875,
                -0.0294342041015625,
                0.01303863525390625,
                -0.041961669921875,
                0.01190948486328125,
                -0.026580810546875,
                -0.01318359375,
                0.0201873779296875,
                -0.0079803466796875,
                0.049072265625,
                0.007053375244140625,
                -0.0228729248046875,
                -0.01059722900390625,
                0.0162506103515625,
                0.0119476318359375,
                0.01300048828125,
                -0.01513671875,
                -0.01200103759765625,
                0.00505828857421875,
                -0.002895355224609375,
                -0.0078125,
                -0.0156402587890625,
                0.0181884765625,
                0.031402587890625,
                0.026580810546875,
                0.024932861328125,
                0.005977630615234375,
                0.0045013427734375,
                0.006557464599609375,
                0.00489044189453125,
                0.01374053955078125,
                -0.018798828125,
                0.0289459228515625,
                0.0151519775390625,
                -0.00893402099609375,
                0.0021877288818359375,
                0.00302886962890625,
                0.019317626953125,
                -0.00833892822265625,
                -0.0037212371826171875,
                -0.003971099853515625,
                -0.0283203125,
                0.036376953125,
                0.00983428955078125,
                -0.0148773193359375,
                0.0205230712890625,
                -0.004730224609375,
                0.008209228515625,
                -0.001407623291015625,
                0.036895751953125,
                0.0153656005859375,
                -0.0212860107421875,
                -0.01861572265625,
                -0.001026153564453125,
                -0.032135009765625,
                0.0195770263671875,
                0.02093505859375,
                -0.07769775390625,
                0.005336761474609375,
                -0.0255126953125,
                0.05316162109375,
                0.0012454986572265625,
                -0.01287078857421875,
                0.02685546875,
                0.0214385986328125,
                -0.0169677734375,
                0.014007568359375,
                -0.0167083740234375,
                -0.01560211181640625,
                0.016571044921875,
                -0.0195770263671875,
                -0.01177978515625,
                0.0305633544921875,
                0.03204345703125,
                0.00945281982421875,
                -0.0033721923828125,
                -0.0300140380859375,
                -0.012481689453125,
                0.00681304931640625,
                0.021484375,
                0.0030803680419921875,
                -0.032318115234375,
                0.08123779296875,
                0.0302886962890625,
                -0.06915283203125,
                0.0406494140625,
                0.002986907958984375,
                -0.03009033203125,
                -0.0242767333984375,
                0.01126861572265625,
                0.031280517578125,
                -0.004425048828125,
                -0.0037784576416015625,
                -0.08502197265625,
                -0.033172607421875,
                -0.0160675048828125,
                -0.02117919921875,
                -0.005458831787109375,
                0.026092529296875,
                -0.06427001953125,
                -0.0188446044921875,
                -0.042572021484375,
                0.038970947265625,
                -0.0031948089599609375,
                0.0027256011962890625,
                0.0092010498046875,
                0.0109100341796875,
                0.027618408203125,
                -0.0022220611572265625,
                0.01641845703125,
                0.043548583984375,
                0.0253143310546875,
                0.049163818359375,
                0.005039215087890625,
                -0.003826141357421875,
                0.004383087158203125,
                0.00634002685546875,
                -0.03656005859375,
                -0.039703369140625,
                0.00830078125,
                -0.0203857421875,
                0.01751708984375,
                -0.005588531494140625,
                0.030487060546875,
                -0.0048370361328125,
                0.004741668701171875,
                0.014495849609375,
                -0.015777587890625,
                0.046722412109375,
                0.0732421875,
                0.0291900634765625,
                -0.042572021484375,
                0.015472412109375,
                0.02117919921875,
                -0.007114410400390625,
                -0.0362548828125,
                0.01641845703125,
                -0.0235443115234375,
                -0.0308074951171875,
                -0.0279693603515625,
                0.058319091796875,
                -0.0340576171875,
                -0.0283660888671875,
                -0.0178070068359375,
                -0.0003132820129394531,
                -0.2015380859375,
                0.041839599609375,
                -0.0142669677734375,
                -0.03375244140625,
                -0.0009174346923828125,
                0.003025054931640625,
                -0.02105712890625,
                -0.0312347412109375,
                -0.01381683349609375,
                -0.042327880859375,
                0.0233306884765625,
                -0.0491943359375,
                0.0190277099609375,
                0.010101318359375,
                -0.028594970703125,
                -0.008758544921875,
                0.038787841796875,
                -0.0282135009765625,
                0.039794921875,
                -0.042724609375,
                -0.0300140380859375,
                0.006664276123046875,
                0.003421783447265625,
                0.015533447265625,
                -0.01068878173828125,
                0.0077667236328125,
                -0.040924072265625,
                0.0223846435546875,
                0.0167999267578125,
                0.021697998046875,
                0.015899658203125,
                0.021026611328125,
                -0.0252227783203125,
                0.024383544921875,
                0.01546478271484375,
                0.01424407958984375,
                0.0294647216796875,
                -0.0203857421875,
                -0.0098876953125,
                -0.021881103515625,
                -0.03326416015625,
                0.0185394287109375,
                -0.0168609619140625,
                0.01214599609375,
                0.008758544921875,
                -0.06414794921875,
                -0.01177215576171875,
                -0.0111083984375,
                -0.01367950439453125,
                0.0352783203125,
                0.007534027099609375,
                -0.023223876953125,
                0.006557464599609375,
                -0.0011339187622070312,
                -0.025482177734375,
                -0.0205535888671875,
                0.0191192626953125,
                0.046630859375,
                0.001323699951171875,
                0.03570556640625,
                -0.01087188720703125,
                -0.033203125,
                0.01885986328125,
                0.041046142578125,
                -0.0025997161865234375,
                -0.0167388916015625,
                -0.038238525390625,
                0.01458740234375,
                0.004459381103515625,
                0.051239013671875,
                0.006855010986328125,
                0.01262664794921875,
                0.0158233642578125,
                -0.0034198760986328125,
                0.0260772705078125,
                0.004665374755859375,
                -0.046905517578125,
                -0.06109619140625,
                -0.047515869140625,
                -0.07275390625,
                0.0211029052734375,
                0.0243377685546875,
                0.006969451904296875,
                -0.02691650390625,
                -0.00801849365234375,
                -0.00046896934509277344,
                -0.019866943359375,
                0.0269775390625,
                0.00817108154296875,
                0.275146484375,
                0.0997314453125,
                -0.046966552734375,
                -0.0115203857421875,
                0.0013599395751953125,
                -0.0049591064453125,
                -0.00020563602447509766,
                0.0001285076141357422,
                0.0092315673828125,
                -0.034332275390625,
                -0.018096923828125,
                0.004993438720703125,
                -0.00409698486328125,
                -0.006587982177734375,
                -0.0195465087890625,
                0.036895751953125,
                0.0027027130126953125,
                0.0024280548095703125,
                0.059783935546875,
                -0.01396942138671875,
                0.045501708984375,
                -0.0645751953125,
                -0.01407623291015625,
                0.031982421875,
                0.01226043701171875,
                -0.01557159423828125,
                -0.033233642578125,
                -0.001987457275390625,
                -0.039093017578125,
                0.022186279296875,
                -0.0162811279296875,
                -0.00988006591796875,
                -0.0217437744140625,
                0.024139404296875,
                0.0143585205078125,
                -0.02191162109375,
                0.029632568359375,
                -0.01488494873046875,
                0.0301513671875,
                0.00817108154296875,
                -0.05487060546875,
                -0.03814697265625,
                -0.01123809814453125,
                -0.03515625,
                -0.016571044921875,
                0.01922607421875,
                -0.03759765625,
                0.00946807861328125,
                -0.0307159423828125,
                -0.0019626617431640625,
                0.0038604736328125,
                0.003376007080078125,
                0.01010894775390625,
                -0.008819580078125,
                0.0291290283203125,
                -0.044952392578125,
                0.0194091796875,
                0.0264129638671875,
                0.0035762786865234375,
                0.042510986328125,
                0.042510986328125,
                0.0304107666015625,
                -0.018646240234375,
                -0.054229736328125,
                0.04083251953125,
                0.0262603759765625,
                -0.0191802978515625,
                -0.005962371826171875,
                0.06622314453125,
                0.0291748046875,
                -0.00730133056640625,
                -0.04632568359375,
                -0.0174560546875,
                -0.0202789306640625,
                0.039398193359375,
                0.033935546875,
                0.0178070068359375,
                0.014678955078125,
                0.01482391357421875,
                -0.00572967529296875,
                0.01812744140625,
                0.006378173828125,
                -0.0177459716796875,
                -0.00887298583984375,
                0.0018167495727539062,
                -0.01023101806640625,
                0.0819091796875,
                0.04937744140625,
                -0.0245208740234375,
                -0.006671905517578125,
                0.02978515625,
                -0.026153564453125,
                -0.058990478515625,
                0.01055145263671875,
                0.0261993408203125,
                -0.060546875,
                0.013275146484375,
                0.0079193115234375,
                -0.045684814453125,
                -0.01561737060546875,
                0.0266876220703125,
                -0.05609130859375,
                0.006809234619140625,
                -9.381771087646484e-05,
                0.045501708984375,
                -0.0087432861328125,
                -0.0263214111328125,
                0.04144287109375,
                -0.031585693359375,
                0.0286712646484375,
                0.0275115966796875,
                0.0004813671112060547,
                0.043792724609375,
                0.02532958984375,
                -0.024200439453125,
                0.005229949951171875,
                0.0205078125,
                -0.048675537109375,
                0.04443359375,
                -0.01448822021484375,
                -0.012664794921875,
                -0.0360107421875,
                -0.005039215087890625,
                -0.035614013671875,
                0.0116119384765625,
                -0.0193939208984375,
                0.002132415771484375,
                0.023651123046875,
                -0.018096923828125,
                0.04754638671875,
                0.0307769775390625,
                -0.00910186767578125,
                -0.04547119140625,
                -0.002223968505859375,
                0.04376220703125,
                0.00363922119140625,
                -0.025238037109375,
                0.01129150390625,
                -0.03753662109375,
                -0.0103302001953125,
                0.039276123046875,
                0.0013399124145507812,
                0.006519317626953125,
                -0.0092010498046875,
                -0.01471710205078125,
                -0.005859375,
                -0.05462646484375,
                -0.01078033447265625,
                -0.032073974609375,
                -0.0069732666015625,
                0.022491455078125,
                -0.015167236328125,
                -0.0224456787109375,
                -0.028594970703125,
                -0.0011682510375976562,
                0.06451416015625,
                0.0014400482177734375,
                -0.030670166015625,
                0.02459716796875,
                -0.04095458984375,
                0.04498291015625,
                0.00951385498046875,
                0.003662109375,
                0.0679931640625,
                0.0260772705078125,
                0.057220458984375,
                -0.0269317626953125,
                0.036895751953125,
                0.0278167724609375,
                -0.006832122802734375,
                -0.0300445556640625,
                0.02703857421875,
                0.01424407958984375,
                -0.00632476806640625,
                0.060577392578125,
                0.059814453125,
                -0.025482177734375,
                -0.0224761962890625,
                -0.01953125,
                -0.00402069091796875,
                0.012847900390625,
                0.037261962890625,
                0.01242828369140625,
                0.01226806640625,
                0.01049041748046875,
                -0.0031719207763671875,
                0.0298919677734375,
                -0.0286712646484375,
                -0.0018777847290039062,
                0.005290985107421875,
                -0.00785064697265625,
                0.130615234375,
                0.040313720703125,
                -0.01136016845703125,
                0.003742218017578125,
                -0.016571044921875,
                -0.02496337890625,
                0.04742431640625,
                -0.019622802734375,
                -0.01279449462890625,
                -0.04571533203125,
                -0.030731201171875,
                0.00833892822265625,
                0.042633056640625,
                -0.002727508544921875,
                -0.006259918212890625,
                -0.03985595703125,
                0.00145721435546875,
                0.0064849853515625,
                -0.038055419921875,
                0.0012884140014648438,
                0.007785797119140625,
                0.0230255126953125,
                -0.0305328369140625,
                -0.026580810546875,
                -0.0261077880859375,
                -0.0022449493408203125,
                -0.0264129638671875,
                -0.035675048828125,
                0.0247955322265625,
                0.00516510009765625,
                -0.0166778564453125,
                0.041168212890625,
                -0.0791015625,
                -0.028350830078125,
                -0.0080108642578125,
                0.0094757080078125,
                -0.01458740234375,
                -0.0245361328125,
                0.0033969879150390625,
                0.00983428955078125,
                0.0430908203125,
                -0.004734039306640625,
                -0.028106689453125,
                -0.00908660888671875,
                0.0253143310546875,
                0.056915283203125,
                -0.01263427734375,
                -0.005645751953125,
                -0.0004401206970214844,
                0.01806640625,
                -0.007442474365234375,
                0.031463623046875,
                0.0021038055419921875,
                0.00086212158203125,
                -0.007549285888671875,
                0.00893402099609375,
                0.00875091552734375,
                0.0060882568359375,
                0.05987548828125,
                0.00582122802734375,
                -0.05206298828125,
                -0.03753662109375,
                -0.00608062744140625,
                -0.0160980224609375,
                0.007518768310546875,
                -0.0173492431640625,
                -0.04339599609375,
                0.00423431396484375,
                -0.0087890625,
                -0.00934600830078125,
                -0.006404876708984375,
                -0.042694091796875,
                -0.021484375,
                -0.01534271240234375,
                -0.03863525390625,
                0.003177642822265625,
                -0.05218505859375,
                0.009002685546875,
                -0.06524658203125,
                0.027313232421875,
                -0.0245208740234375,
                -0.041717529296875,
                0.0069122314453125,
                -0.016021728515625,
                0.0081787109375,
                -0.0189056396484375,
                0.048004150390625,
                0.004344940185546875,
                0.0255279541015625,
                -0.0223388671875,
                0.0114593505859375,
                0.006305694580078125,
                0.00574493408203125,
                -0.0230712890625,
                -0.009063720703125,
                0.0181884765625,
                0.00681304931640625,
                -0.01061248779296875,
                -0.0201263427734375,
                0.033233642578125,
                -0.037109375,
                0.0107421875,
                0.00858306884765625,
                -0.034912109375,
                -0.0017518997192382812,
                -0.01433563232421875,
                -0.01425933837890625,
                -0.0333251953125,
                -0.0251617431640625,
                -0.03350830078125,
                0.01358795166015625,
                -0.04595947265625,
                -0.0219573974609375,
                -0.004901885986328125,
                0.01543426513671875,
                -0.01494598388671875,
                -0.046966552734375,
                -0.00992584228515625,
                -0.06195068359375,
                -0.004444122314453125,
                -0.06939697265625,
                0.0291290283203125,
                -0.0389404296875,
                -0.02581787109375,
                -0.04296875,
                0.0310516357421875,
                -0.014007568359375,
                -0.040618896484375,
                0.0160064697265625,
                0.0009160041809082031,
                0.032562255859375,
                -0.01514434814453125,
                -0.02374267578125,
                -0.03570556640625,
                0.06988525390625,
                0.03424072265625,
                0.0254364013671875,
                0.0241546630859375,
                0.03912353515625,
                -0.035064697265625,
                0.039825439453125,
                -0.0274810791015625,
                0.00887298583984375,
                0.053802490234375,
                0.020172119140625,
                0.0002505779266357422,
                0.00168609619140625,
                -0.021453857421875,
                -0.027099609375,
                0.00768280029296875,
                0.0306243896484375,
                0.021514892578125,
                -0.032501220703125,
                0.016448974609375,
                -0.106689453125,
                -0.0302886962890625,
                0.0618896484375,
                -0.00965118408203125,
                0.05322265625,
                0.00995635986328125,
                0.01049041748046875,
                0.005023956298828125,
                -0.0234375,
                0.005428314208984375,
                0.004428863525390625,
                0.03631591796875,
                -0.00019109249114990234,
                0.0160369873046875,
                -0.03375244140625,
                -0.019439697265625,
                0.04522705078125,
                -0.0285186767578125,
                -0.028472900390625,
                -0.04730224609375,
                -0.00836181640625,
                0.031463623046875,
                -0.0005860328674316406,
                -0.014678955078125,
                0.0138092041015625,
                -0.004550933837890625,
                0.0599365234375,
                -0.00823974609375,
                -0.00762176513671875,
                -0.0070343017578125,
                0.0005450248718261719,
                0.004032135009765625,
                -0.001300811767578125,
                -0.0010395050048828125,
                0.04852294921875,
                -0.0265350341796875,
                -0.0311126708984375,
                -0.01523590087890625,
                -0.0171661376953125,
                -0.00830078125,
                0.041351318359375,
                -0.0400390625,
                0.0265350341796875,
                0.0039825439453125,
                -0.01136016845703125,
                -0.013336181640625,
                -0.0196533203125,
                -0.0017004013061523438,
                -0.00704193115234375,
                -0.1395263671875,
                -0.0138397216796875,
                0.02239990234375,
                0.00800323486328125,
                -0.041534423828125,
                0.00353240966796875,
                -0.003849029541015625,
                -0.0145263671875,
                -0.0004649162292480469,
                -0.0131683349609375,
                -0.034881591796875,
                0.041107177734375,
                0.0396728515625,
                0.0218658447265625,
                -0.07666015625,
                0.00823974609375,
                0.021820068359375,
                -0.01495361328125,
                0.01015472412109375,
                -0.0285797119140625,
                0.0026340484619140625,
                -0.0247344970703125,
                0.04827880859375,
                0.038421630859375,
                -0.0026874542236328125,
                0.0338134765625,
                -0.034423828125,
                0.002685546875,
                -0.0543212890625,
                -0.01861572265625,
                0.048553466796875,
                0.017669677734375,
                0.0073089599609375,
                0.020416259765625,
                0.044036865234375,
                -0.03271484375,
                0.01483917236328125,
                -0.019927978515625,
                0.08270263671875,
                -0.0016450881958007812,
                0.03179931640625,
                0.0211029052734375,
                0.01451873779296875,
                0.000995635986328125,
                0.004169464111328125,
                0.055389404296875,
                -0.031280517578125,
                -0.00568389892578125,
                -0.02099609375,
                -0.03985595703125,
                -0.00838470458984375,
                -0.0113983154296875,
                0.041534423828125,
                0.0269775390625,
                0.007720947265625,
                0.052154541015625,
                -0.01291656494140625,
                -0.032806396484375,
                0.032257080078125,
                0.03656005859375,
                0.0184326171875,
                -0.043670654296875,
                0.0008978843688964844,
                0.003673553466796875,
                -0.003726959228515625,
                0.005741119384765625,
                -0.01018524169921875,
                0.044952392578125,
                0.0223846435546875,
                -0.0016756057739257812,
                0.01442718505859375,
                -0.06927490234375,
                -0.02264404296875,
                -0.060821533203125,
                -0.00044655799865722656,
                -0.0290679931640625,
                0.0621337890625,
                0.009033203125,
                -0.043426513671875,
                -0.09136962890625,
                0.0576171875,
                -0.015869140625,
                -0.051788330078125,
                -0.01242828369140625,
                -0.03302001953125,
                -0.01390838623046875,
                -0.01343536376953125,
                -0.0018644332885742188,
                -0.0284576416015625,
                0.00833892822265625,
                0.00817108154296875,
                -0.0234375,
                0.0188140869140625,
                -0.0180816650390625,
                -0.03167724609375,
                -0.0258026123046875,
                0.01506805419921875,
                0.005832672119140625,
                -0.00914764404296875,
                0.019805908203125,
                -0.021575927734375,
                -0.01526641845703125,
                0.01485443115234375,
                0.0096893310546875,
                -0.037353515625,
                0.062408447265625,
                -0.020172119140625,
                0.00923919677734375,
                -0.0263671875,
                -0.032379150390625,
                0.048126220703125,
                0.004222869873046875,
                -0.024169921875,
                -0.0189056396484375,
                -0.0256195068359375,
                0.050445556640625,
                0.0087890625,
                -0.0172271728515625,
                0.0026378631591796875,
                -0.0185699462890625,
                -0.01117706298828125,
                -0.034820556640625,
                -0.035186767578125,
                0.0010957717895507812,
                0.006893157958984375,
                0.0017786026000976562,
                0.04974365234375,
                0.010009765625,
                -0.022796630859375,
                0.005523681640625,
                0.02154541015625,
                -0.07220458984375,
                0.033203125,
                0.01300048828125,
                -0.0110931396484375,
                -0.0296478271484375,
                -0.0213165283203125,
                0.00212860107421875,
                -0.06524658203125,
                -0.051513671875,
                0.0300445556640625,
                0.0298919677734375,
                -0.0246124267578125,
                -0.0196075439453125,
                -0.01763916015625,
                -0.0242767333984375,
                0.033935546875,
                0.01027679443359375,
                -0.047271728515625,
                -0.03741455078125,
                -0.0030536651611328125,
                0.03302001953125,
                0.0079498291015625,
                0.034423828125,
                0.0310516357421875,
                -0.053497314453125,
                0.062744140625,
                0.0164337158203125,
                0.0161895751953125,
                0.02923583984375,
                -0.0004324913024902344,
                -0.01226806640625,
                -0.01053619384765625,
                -0.016632080078125,
                -0.0058135986328125,
                0.03448486328125,
                -0.0255584716796875,
                0.0214996337890625,
                -0.0211944580078125,
                0.006908416748046875,
                0.01250457763671875,
                0.0010509490966796875,
                0.006122589111328125,
                -0.00905609130859375,
                -0.0082855224609375,
                -0.036041259765625,
                0.01085662841796875,
                -0.01010894775390625,
                0.05096435546875,
                -0.03363037109375,
                0.0073394775390625,
                0.00627899169921875,
                0.031585693359375,
                -0.0018033981323242188,
                0.01436614990234375,
                -0.01171112060546875,
                0.01329803466796875,
                0.002559661865234375,
                -0.052734375,
                -0.050018310546875,
                0.0194854736328125,
                -0.0266876220703125,
                -0.033935546875,
                -0.01084136962890625,
                0.00870513916015625,
                -0.05657958984375,
                0.00478363037109375,
                0.0137786865234375,
                -0.00785064697265625,
                -0.00617218017578125,
                -0.0305328369140625,
                -0.01308441162109375,
                -0.035888671875,
                -0.0017862319946289062,
                -0.0242919921875,
                -0.006488800048828125,
                0.00769805908203125,
                0.034698486328125,
                -0.039703369140625,
                -0.038848876953125,
                -0.0692138671875,
                -0.0281982421875,
                0.035552978515625,
                0.06536865234375,
                0.006198883056640625,
                -0.00705718994140625,
                0.0257415771484375,
                0.042633056640625,
                -0.0188446044921875,
                -0.0216217041015625,
                0.027130126953125,
                0.08154296875,
                -0.0227508544921875,
                -0.035919189453125,
                0.03961181640625,
                0.0038433074951171875,
                -0.02972412109375,
                0.00225067138671875,
                -0.00017952919006347656,
                -0.07421875,
                0.0120391845703125,
                -0.034698486328125,
                0.0265960693359375,
                0.0113525390625,
                0.022003173828125,
                -0.01654052734375,
                -0.0233001708984375,
                0.016357421875,
                -0.05487060546875,
                -0.0005745887756347656,
                0.03411865234375,
                0.0102691650390625,
                -0.0069580078125,
                0.007549285888671875
            ],
            "sparse_vector": {
                "4": 0.10089111328125,
                "6": 0.111083984375,
                "30": 0.0921630859375,
                "37": 0.0141143798828125,
                "264": 0.1043701171875,
                "267": 0.234375,
                "465": 0.014312744140625,
                "468": 0.08538818359375,
                "470": 0.10162353515625,
                "487": 0.11480712890625,
                "514": 0.0143585205078125,
                "562": 0.01385498046875,
                "568": 0.2095947265625,
                "573": 0.01425933837890625,
                "575": 0.01430511474609375,
                "886": 0.06622314453125,
                "887": 0.01280975341796875,
                "1034": 0.01438140869140625,
                "1040": 0.01428985595703125,
                "1064": 0.10125732421875,
                "1102": 0.03558349609375,
                "1107": 0.01454925537109375,
                "1130": 0.0128173828125,
                "1198": 0.0256500244140625,
                "1317": 0.0144195556640625,
                "1344": 0.014373779296875,
                "1403": 0.011199951171875,
                "1420": 0.04974365234375,
                "1493": 0.0222930908203125,
                "1553": 0.0987548828125,
                "1580": 0.07666015625,
                "1677": 0.0143280029296875,
                "1726": 0.09210205078125,
                "1801": 0.01447296142578125,
                "1826": 0.03662109375,
                "1844": 0.013824462890625,
                "1906": 0.061309814453125,
                "1955": 0.064453125,
                "1965": 0.01209259033203125,
                "2003": 0.011260986328125,
                "2100": 0.0255584716796875,
                "2128": 0.0693359375,
                "2308": 0.01436614990234375,
                "2398": 0.01082611083984375,
                "2657": 0.02203369140625,
                "2672": 0.12457275390625,
                "2698": 0.0278167724609375,
                "2797": 0.0270233154296875,
                "2825": 0.0584716796875,
                "2830": 0.04443359375,
                "2969": 0.07177734375,
                "2996": 0.0202789306640625,
                "3270": 0.01337432861328125,
                "3302": 0.0523681640625,
                "3452": 0.04266357421875,
                "3508": 0.17626953125,
                "3524": 0.0132598876953125,
                "3586": 0.03936767578125,
                "3715": 0.14111328125,
                "3759": 0.0251312255859375,
                "3803": 0.1259765625,
                "3891": 0.0947265625,
                "3924": 0.09088134765625,
                "3933": 0.08880615234375,
                "4121": 0.0379638671875,
                "4150": 0.0211334228515625,
                "4280": 0.0142974853515625,
                "4290": 0.004650115966796875,
                "4467": 0.01140594482421875,
                "4617": 0.056396484375,
                "4695": 0.05267333984375,
                "4723": 0.0212249755859375,
                "4766": 0.011871337890625,
                "5133": 0.037841796875,
                "5260": 0.141357421875,
                "5292": 0.0183563232421875,
                "5364": 0.1680908203125,
                "5511": 0.0980224609375,
                "5680": 0.11492919921875,
                "5714": 0.1824951171875,
                "5858": 0.01338958740234375,
                "5873": 0.014068603515625,
                "5958": 0.05633544921875,
                "6062": 0.07843017578125,
                "6071": 0.002635955810546875,
                "6082": 0.08721923828125,
                "6271": 0.0814208984375,
                "6349": 0.1083984375,
                "6717": 0.11968994140625,
                "6728": 0.031585693359375,
                "6986": 0.007312774658203125,
                "7048": 0.06756591796875,
                "7149": 0.1427001953125,
                "7234": 0.2025146484375,
                "7318": 0.04498291015625,
                "7800": 0.097412109375,
                "8042": 0.029541015625,
                "9002": 0.0999755859375,
                "9095": 0.09674072265625,
                "9243": 0.1153564453125,
                "9358": 0.251708984375,
                "9524": 0.09259033203125,
                "9586": 0.0274200439453125,
                "9909": 0.07525634765625,
                "10132": 0.059967041015625,
                "10346": 0.003421783447265625,
                "10406": 0.048431396484375,
                "10721": 0.13427734375,
                "10732": 0.252685546875,
                "11152": 0.0126800537109375,
                "11158": 0.0245513916015625,
                "11359": 0.056671142578125,
                "11415": 0.01160430908203125,
                "11599": 0.0171356201171875,
                "12392": 0.1029052734375,
                "12590": 0.060302734375,
                "12790": 0.06951904296875,
                "12821": 0.0221405029296875,
                "13012": 0.085205078125,
                "13447": 0.1270751953125,
                "13525": 0.026885986328125,
                "14250": 0.01490020751953125,
                "14635": 0.106689453125,
                "15041": 0.1385498046875,
                "15373": 0.09417724609375,
                "15475": 0.01806640625,
                "15900": 0.060028076171875,
                "17178": 0.1429443359375,
                "17599": 0.01059722900390625,
                "17844": 0.006168365478515625,
                "18220": 0.0196075439453125,
                "18542": 0.0787353515625,
                "18677": 0.02288818359375,
                "19312": 0.1865234375,
                "19390": 0.0372314453125,
                "19431": 0.0750732421875,
                "19752": 0.1246337890625,
                "20373": 0.12030029296875,
                "22292": 0.0124359130859375,
                "22773": 0.041290283203125,
                "23398": 0.017120361328125,
                "23690": 0.2352294921875,
                "24257": 0.0012836456298828125,
                "24789": 0.10400390625,
                "25786": 0.0977783203125,
                "25881": 0.096923828125,
                "27644": 0.1097412109375,
                "28226": 0.0186004638671875,
                "29026": 0.03448486328125,
                "29085": 0.040496826171875,
                "29100": 0.012664794921875,
                "30828": 0.10894775390625,
                "30832": 0.08013916015625,
                "30983": 0.07086181640625,
                "31023": 0.2447509765625,
                "31562": 0.2093505859375,
                "31585": 0.1302490234375,
                "32004": 0.023193359375,
                "34101": 0.122314453125,
                "34484": 0.020843505859375,
                "35303": 0.0694580078125,
                "35368": 0.09783935546875,
                "35726": 0.032135009765625,
                "36247": 0.0194244384765625,
                "36551": 0.29052734375,
                "36602": 0.10455322265625,
                "36852": 0.1192626953125,
                "37084": 0.1297607421875,
                "37882": 0.01425933837890625,
                "39031": 0.034332275390625,
                "39904": 0.08990478515625,
                "40870": 0.018341064453125,
                "42465": 0.021087646484375,
                "42510": 0.1002197265625,
                "42986": 0.07769775390625,
                "45531": 0.01335906982421875,
                "46222": 0.03204345703125,
                "46453": 0.102783203125,
                "46530": 0.01107025146484375,
                "49163": 0.10198974609375,
                "49340": 0.04046630859375,
                "51283": 0.07989501953125,
                "53564": 0.143310546875,
                "55425": 0.1480712890625,
                "57189": 0.12066650390625,
                "59880": 0.060882568359375,
                "60068": 0.15380859375,
                "60988": 0.230224609375,
                "61480": 0.01428985595703125,
                "63851": 0.15087890625,
                "64218": 0.11651611328125,
                "65701": 0.039520263671875,
                "70021": 0.13671875,
                "70884": 0.10906982421875,
                "71077": 0.10968017578125,
                "73619": 0.382080078125,
                "74171": 0.1258544921875,
                "76836": 0.1265869140625,
                "79921": 0.08013916015625,
                "82436": 0.0750732421875,
                "85682": 0.1251220703125,
                "86708": 0.0469970703125,
                "90586": 0.0258026123046875,
                "92181": 0.1568603515625,
                "94010": 0.1304931640625,
                "95570": 0.032012939453125,
                "101458": 0.0089111328125,
                "101523": 0.06201171875,
                "102852": 0.1229248046875,
                "103612": 0.07666015625,
                "106221": 0.092041015625,
                "106753": 0.20654296875,
                "107113": 0.133544921875,
                "107283": 0.0394287109375,
                "111078": 0.10015869140625,
                "116529": 0.1644287109375,
                "121072": 0.00905609130859375,
                "124115": 0.1617431640625,
                "126709": 0.01483154296875,
                "127700": 0.018768310546875,
                "128772": 0.11273193359375,
                "129009": 0.199951171875,
                "131582": 0.0789794921875,
                "132372": 0.1658935546875,
                "138138": 0.032470703125,
                "141121": 0.00946044921875,
                "143678": 0.01396942138671875,
                "144912": 0.0096893310546875,
                "151161": 0.146728515625,
                "157623": 0.134033203125,
                "170412": 0.08514404296875,
                "182942": 0.13037109375,
                "188814": 0.1630859375,
                "189657": 0.07720947265625,
                "193055": 0.0921630859375,
                "195668": 0.052459716796875,
                "210457": 0.21240234375,
                "212566": 0.09539794921875,
                "214820": 0.11651611328125,
                "215730": 0.0306549072265625,
                "223449": 0.114501953125,
                "229850": 0.10284423828125,
                "232269": 0.129638671875,
                "238387": 0.02752685546875,
                "238500": 0.11297607421875,
                "243403": 0.0166015625,
                "243417": 0.04345703125,
                "243615": 0.01276397705078125,
                "243808": 0.01265716552734375,
                "243909": 0.10113525390625,
                "243921": 0.00646209716796875,
                "244026": 0.045867919921875,
                "244081": 0.08099365234375,
                "244121": 0.1099853515625,
                "244244": 0.0119476318359375,
                "244315": 0.09466552734375,
                "244317": 0.1455078125,
                "244413": 0.1888427734375,
                "244708": 0.1932373046875,
                "244852": 0.08721923828125,
                "245314": 0.206298828125,
                "245360": 0.086181640625,
                "245691": 0.1380615234375,
                "245743": 0.1500244140625,
                "246157": 0.15771484375,
                "246802": 0.1295166015625,
                "247969": 0.08099365234375,
                "248680": 0.16455078125,
                "248726": 0.032928466796875,
                "248943": 0.0992431640625,
                "249344": 0.1312255859375,
                "249962": 0.1806640625
            },
            "chunk_id": 465017001340914266
        }
    ]
}

    knowledge_graph_node = KnowLedgeGraphNode()

    knowledge_graph_node.process(mock_state)


if __name__ == "__main__":
    test_kg_extraction()
