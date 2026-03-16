from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError

from pathlib import Path
import subprocess
import time

class PdfToMdNode(BaseNode):
    """
    pdf转换md节点
    """
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        Args:
            self:
            state:

        Returns:

        """
        # 1. 对参数校验
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        # 2. 利用MinerU工具解析pdf成为md
        processed_code = self._execute_mineru(import_file_path, file_dir_path)
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF文件失败", self.name)

        # 3. 获取md的path
        md_file_path = self._get_md_path(import_file_path, file_dir_path)
        # 4.更新state字典的md_path
        state["md_path"] = str(md_file_path)
        # 5.返回state
        return state



    def _validate_state_inputs_path(self, state: ImportGraphState) -> tuple[Path, Path]:
        """
        Args:
            state:该节点收到的状态

        Returns:


        """
        self.log_step("step1", "对状态的路径输入参数做校验")

        # 1. 获取输入pdf的文件路径
        import_file_path = state.get("import_file_path", "")
        # 2. 获取解析后的输出路径
        file_dir = state.get("file_dir", "")
        # 3. 校验输出的文件路径(非空判断不判断是否有效)
        if not import_file_path:
            raise ValidationError("解析的文件路径不能为空", self.name)
        
        # 4. 用Path标准化
        import_file_path_opj = Path(import_file_path)
        # 5. 校验是否是一个真实的路径（也就是是否有效）
        if not import_file_path_opj.exists():
            raise FileProcessingError("解析的文件路径不存在", self.name)
        # 6. 校验输出目录是否为空
        if not file_dir:
            # 给一个默认目录兜底
            file_dir = import_file_path_opj.parent
        
        # 7. 返回输出目录标准化
        file_dir_path_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径：{import_file_path}")
        self.logger.info(f"输出的目录：{file_dir}")

        # 8. 返回输出文件以及输出目录的标准path
        return import_file_path_opj, file_dir_path_obj
    
    def _execute_mineru(self, import_file_path: Path, file_dir_path: Path):
        """
        Args:
            import_file_path: 解析的文件路径
            file_dir_path: 解析后的文件目录

        Returns:
            mineru -p <input_path> -o <output_path> --source local
        """
        ## 执行命令行

        self.log_step("step2", "利用MinerU工具解析pdf")

        # 1. 构建命令行
        cmd = [
            "mineru",
            "-p",
            str(import_file_path),
            "-o",
            str(file_dir_path),
            "--source",
            "local"
        ]

        process_start_time = time.time()
        # 2. 执行命令行(子进程执行命令行) 自动读取到主进程的环境变量
        proc = subprocess.Popen(
                args = cmd, 
                # 我要将mineru产生的日志打印到我的日志中
                stdout = subprocess.PIPE,
                # 接受错误日志， 并将其合并到stdout中
                stderr=subprocess.STDOUT,
                # 像乱码问题（不认识的日志替换掉） 一般替换为问号（？）或者空心矩形（□）
                errors = "replace",
                # 接受数据
                text=True, # 保证输出内容为字符串， 不是字节
                encoding="gbk",
                # 实时打印日志
                bufsize=1, # 按行缓冲区 只要缓冲区一行满了就输出
            )
        
        # 3. 获取日志信息
        for line in proc.stdout:
            self.logger.info(f"执行mineru产生的日志:{line}")

        # 4. 等待子进程执行完毕(主进程等待子进程做完)
        processed_code = proc.wait() # 做完了状态码(processed_code)就是0
        process_end_time = time.time()

        if processed_code == 0:
            self.logger.info(f"MinerU成功解析PDF文件:{import_file_path.name}, 耗时:{process_end_time - process_start_time:.2f}")
        else:
            self.logger.info(f"MinerU解析PDF文件:{import_file_path.name}失败")

        # 5. 返回状态码
        return processed_code
    
    def _get_md_path(self, import_file_path: Path, file_dir_path: Path) -> Path:
        """
        Args:
            import_file_path: 解析的文件路径
            file_dir_path: 解析后的文件目录

        Returns:
            md_file_path: 解析后的md文件路径
        """
        # 1. 从import_file_path中获取文件名（不包含扩展名）
        file_name_without_ext = import_file_path.stem
        # 2. 构建md文件路径
        md_file_path = file_dir_path / file_name_without_ext / "hybrid_auto" / f"{file_name_without_ext}.md"
        # 3. 返回md文件路径
        return md_file_path



if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = PdfToMdNode()

    pdf_to_md_node_init_state = {
        "import_file_path": r"D:\pycharm\project\shopkeeper_brain\scripts\Original document\高中数学知识点归纳.pdf",
        "file_dir": r"D:\pycharm\project\shopkeeper_brain\scripts\processed"
    }
    pdf_to_md_node.process(pdf_to_md_node_init_state)


