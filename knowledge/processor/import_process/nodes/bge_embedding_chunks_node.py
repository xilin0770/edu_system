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
            # 1.1 提取title  as： “# 1、定积分的概念-2-2”
            title = chunk.get('title')

            # 1.1 提取body
            body = chunk.get('body')

            # 1.2 提取item_name
            # math_concept = chunk.get('math_concept')

            # 1.3 拼接要嵌入的最终内容
            embedding_content = f"{title}\n{body}"

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

    # input_path = base_temp_dir / "chunks.json"
    output_path = base_temp_dir / "chunks_vector.json"

    # 1. 读取上游状态
    # if not input_path.exists():
    #     print(f" 找不到输入文件: {input_path}")

    # with open(input_path, "r", encoding="utf-8") as f:
    #     content = json.load(f)

    # 2. 构建模拟的图状态 (Graph State)
    state = {
        "chunks": [{
            "title": "# 6、参数方程的概念",
            "body": "\n在平面直角坐标系中，如果曲线上任意一点的坐标\n\n$\\mathbf{x},\\mathbf{y}$ 都是某个变数 $\\mathbf{t}$ 的函数 $\\left\\{ \\begin{array}{l}\\mathbf{x} = \\mathbf{f}(\\mathbf{t}),\\\\ \\mathbf{y} = \\mathbf{g}(\\mathbf{t}), \\end{array} \\right.$ 并且对于 $\\mathbf{t}$ 的每一个允许值，由这个方程所确定的点 $\\mathbf{M}(\\mathbf{x},\\mathbf{y})$ 都在这条曲线上，那么这个方程就叫做这条曲线的 参数方\n\n程，联系变数 $x, y$ 的变数 $t$ 叫做参变数，简称参数。相对于参数方程而言，直接给出点的坐标间关系的方程叫 做普通方程。\n",
            "file_title": "万用表的使用",
            "parent_title": "# 6、参数方程的概念",
            "content": "# 6、参数方程的概念\n\n在平面直角坐标系中，如果曲线上任意一点的坐标\n\n$\\mathbf{x},\\mathbf{y}$ 都是某个变数 $\\mathbf{t}$ 的函数 $\\left\\{ \\begin{array}{l}\\mathbf{x} = \\mathbf{f}(\\mathbf{t}),\\\\ \\mathbf{y} = \\mathbf{g}(\\mathbf{t}), \\end{array} \\right.$ 并且对于 $\\mathbf{t}$ 的每一个允许值，由这个方程所确定的点 $\\mathbf{M}(\\mathbf{x},\\mathbf{y})$ 都在这条曲线上，那么这个方程就叫做这条曲线的 参数方\n\n程，联系变数 $x, y$ 的变数 $t$ 叫做参变数，简称参数。相对于参数方程而言，直接给 出点的坐标间关系的方程叫做普通方程。\n",
            "math_concept": "集合 - 集合概念与表示\n函数 - 函数概念与性质\n函数 - 基本初等函数 - 指数函数\n函数 - 基本初等函数 - 对数函数\n函数 - 基本初等函数 - 幂函数\n立体几何 - 立体几何初步\n解析几何 - 平面解析几何初步\n算法 - 算法初步\n统计 - 统计\n概率 - 概率\n函数 - 基本初等函数 - 三角函数\n向量 - 平面向量\n三角函数 - 三角恒等变换\n解三角形 - 解三角形\n数列 - 数列\n不等式 - 不等式\n常用逻辑用语 - 常用逻辑用语\n解析几何 - 圆锥曲线与方程\n微积分 - 导数及其应用\n统计 - 统计案例\n推理与证明 - 推理与证明\n复数 - 数系 的扩充与复数\n算法 - 框图\n立体几何 - 空间向量与立体几何\n计数原理 - 计数原理\n概率 - 随机变量及其分布列\n数学史 - 数学史选讲\n信息安全与密码 - 信息安全与密码\n几何 - 球面上的几何\n代数 - 对称与群\n几何 - 欧拉公式与闭曲面分类\n几何 - 三等分角与数域扩充\n几何 -  几何证明选讲\n代数 - 矩阵与变换\n数列 - 数列与差分\n解析几何 - 坐标系与参数方程\n不等式 - 不等式选讲\n数论 - 初等数论初步\n优选法与试验设计 - 优选法与试验设计初步\n统筹法与图论 - 统筹法与图论初步\n风险与决策 - 风险与决策\n开关电路与布尔代数 - 开关电路与布尔代数"
        }]
    }

    # 3. 触发节点执行
    node_bge_embedding = BgeEmbeddingChunksNode()
    proceed_result = node_bge_embedding.process(state)

    # 4. 结果落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f" 向量生成测试完成！结果已成功备份至:\n{output_path}")
