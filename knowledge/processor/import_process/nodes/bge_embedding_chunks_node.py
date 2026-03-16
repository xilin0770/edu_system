import os, json
from typing import Dict, List, Any
from pathlib import Path
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, EmbeddingError
from knowledge.processor.import_process.config import get_config
from knowledge.utils.bge_me_embedding_util import get_beg_m3_embedding_model

class BgeEmbeddingChunksNode(BaseNode):
    """
    1. 获取所有chunks拼接的向量内容
    2. 批量嵌入chunk的embedding_content:item_name+chunk
    3. 将所有chunk嵌入后的向量值存储到列表中 在返回给下一个节点用
    """
    name = "bge_embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 参数校验
        chunks, config = self._validate_get_inputs(state)

        # 2. 获取批量嵌入的阈值
        embedding_batch_chunk_size = getattr(config, "embedding_batch_size", 16)

        # 3. 准备分批次嵌入（pineline）
        # 待嵌入的所有数据chunks=[1,2,3,4,5,6]
        # 阈值：3
        # 第一批：[1,2,3]
        # 第二批：[4,5,6]

        # 待嵌入的所有数据chunks=[1,2]
        # 阈值：3
        # 第一批：[1,2,3]
        total_length = len(chunks)
        final_chunks = []
        for i in range(0, total_length, embedding_batch_chunk_size):
            batch = chunks[i:i + embedding_batch_chunk_size]
            # 拼接要嵌入的内容 向量嵌入的内容 把嵌入的向量注入到chunk中
            batch_chunks = self._process_batch_chunks(batch, i, total_length)
            final_chunks.extend(batch_chunks)

        # 4. 更新&返回state
        state['chunks'] = final_chunks

        return state
    
    def _process_batch_chunks(self, batch: List[Dict[str, Any]], star_index: int, total_length: int):

        self.log_step("step2", f"开始批量处理chunk嵌入:批次{star_index + 1}-{star_index + len(batch)}")
        # 1. 循环处理所有chunk的要嵌入的内容拼接
        embedding_contents = []
        for _, chunk in enumerate(batch):
            # 1.1 提取content
            content = chunk.get('content')

            # 1.2 提取item_name
            item_name = chunk.get('item_name')

            # 1.3 拼接要嵌入的最终内容
            embedding_content = f"{item_name}\n{content}"

            embedding_contents.append(embedding_content)

        # 2. 批量嵌入
        try:
            bge_m3_model = get_beg_m3_embedding_model()
            embedding_result = bge_m3_model.encode_documents(documents=embedding_contents)

            if not embedding_result:
                self.logger.warning(f"嵌入后的结果不存在...")
                return batch
        except Exception as e:
            self.logger.warning(f"嵌入向量嵌入失败...{str(e)}")
            return batch

        # 3. 循环处理所有chunk的向量以及注入到每一个chunk中
        for index, chunk in enumerate(batch):
            # 3.1 获取稠密向量
            dense_vector = embedding_result['dense'][index].tolist()

            # 3.2 解构csr矩阵&获取稀疏向量
            csr_array = embedding_result['sparse']
            # a) 行索引
            ind_ptr = csr_array.indptr

            # b) 获取行索引的起始值
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index + 1]

            # c) 获取token_id
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()

            # d) 获取权重
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()

            # 3.3 获取稀疏向量
            sparse_vector = dict(zip(token_id, weight))

            # 3.4 注入
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector

        self.logger.info(f"开始批量处理chunk嵌入:批次{star_index + 1}-{star_index + len(batch)}/{total_length}")
        return batch

    def _validate_get_inputs(self, state: ImportGraphState):
        config = get_config()

        self.log_step("step1", "验证输入参数是否存在")

        # 1. 获取chunks
        chunks = state.get("chunks", [])

        # 2. 校验chunks
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunks参数缺失或不是列表类型", self.name)
        
        # 3. 返回
        self.logger.info(f"嵌入的块数:{len(chunks)}")

        return chunks, config
    
if __name__ == '__main__':
    setup_logging()

    base_temp_dir = Path(
        r"D:\pycharm\project\shopkeeper_brain\scripts\processed\高中数学知识点归纳\hybrid_auto")

    input_path = base_temp_dir / "chunks.json"
    output_path = base_temp_dir / "chunks_vector.json"

    # 1. 读取上游状态
    if not input_path.exists():
        print(f" 找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 2. 构建模拟的图状态 (Graph State)
    state = {
        "chunks": content
    }

    # 3. 触发节点执行
    node_bge_embedding = BgeEmbeddingChunksNode()
    proceed_result = node_bge_embedding.process(state)

    # 4. 结果落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f" 向量生成测试完成！结果已成功备份至:\n{output_path}")
