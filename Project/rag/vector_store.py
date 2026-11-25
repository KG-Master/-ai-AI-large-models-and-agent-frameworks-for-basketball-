"""
【知识库写入端 / 索引层（离线侧）】

职责：把 data 目录下的原始文件（txt / pdf）加工成向量，持久化进 Chroma。
本质是一条 ETL 流水线：
    读取文件 -> 文本切分 -> 向量化(embed_model) -> 写入向量库(chroma_db)

运行时机：知识库文件有新增或修改时才需要跑（低频、耗时、可离线执行）。
产出：磁盘上的向量库，供 rag_service.py 在用户每次提问时检索。

注意：本文件只负责"把资料准备好"，不负责"回答用户问题"。
      末尾 __main__ 里的检索调用只是入库后的自检。
"""
import os.path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_handler import chroma_conf
from model.factory import embed_model
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    """知识库索引服务：一个实例 = 一个 Chroma collection 的操作入口。"""

    def __init__(self):
        # 连接持久化的 Chroma 库（不存在则创建）：
        # - collection_name 必须与查询端一致，否则查询端会读到一个空库
        # - embedding_function 必须与建库时一致，换 embedding 模型后旧向量不可复用
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )
        # 文本切分器：参数全部来自 config/chroma.yml
        # chunk_size / chunk_overlap 直接决定检索片段的粒度，属于"写侧"调参
        self.spilter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        # 把向量库包装成"检索器"对象，这是提供给查询端（rag_service.py）的接口。
        # k 表示每次召回多少个分片，只影响查询时的取回数量，不改变库里已有内容。
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        增量、幂等地把 data 目录下的文件写入向量库。

        完整步骤：
            1. 扫描 data 目录，拿到允许类型的文件路径
            2. 计算文件 md5，与 md5.text 记录比对，处理过就跳过
            3. 读取文件（txt/pdf）-> 切分成多个 Document 分片
            4. 按 source 删除该文件的旧分片，再写入新分片（支持文件内容更新）
            5. 记录 md5，避免下次重复入库

        所以重复执行本方法是安全的：不会产生重复数据，也不会漏掉新文件。
        """
        # 先获得知识库的文件，多个文件就是最终获得tuple[str]，元组里是文件的绝对文件路径

        def check_md5_hex(md5_for_check: str):
            # md5 记录文件用于判断"这个文件的内容是否已经入过库"
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False  # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  # md5 处理过

                return False  # md5 没处理过
        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 第 1 步：用文件内容指纹判断是否已入库（内容没变就跳过，省掉 embedding 开销）
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path} 内容已经存在知识库内，跳过加载")
                continue

            try:
                # 第 2 步：加载原始文本（pdf / txt 各走不同 loader）
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过加载")
                    continue

                # 第 3 步：切分。长文档必须切成小分片，否则检索粒度过粗、也撑爆模型上下文
                split_document: list[Document] = self.spilter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过加载")
                    continue

                # 第 4 步：同一文件重新索引前，先删掉它此前写入的全部分片，避免旧内容残留
                self.vector_store.delete(where={"source": {"$eq": path}})

                # 第 5 步：向量化并写入（内部会调用 embed_model，是整条流程最耗时的环节）
                self.vector_store.add_documents(split_document)

                # 第 6 步：记录该文件的 md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path} 加载失败: {str(e)}",exc_info=True)
                continue

if __name__ == '__main__':
    # 独立运行本文件 = 执行一次"建库 + 自检"，与用户问答无关
    vs = VectorStoreService()

    vs.load_document()  # 建库 / 增量更新

    retriever = vs.get_retriever()

    # 自检：直接用一句 query 看能否召回内容，确认入库成功
    res = retriever.invoke("三分线距离")
    for r in res:
        print(r.page_content)
        print("-"*20)
