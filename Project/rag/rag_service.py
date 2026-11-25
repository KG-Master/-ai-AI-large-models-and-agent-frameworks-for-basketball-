'''
【知识库查询端 / 生成层（在线侧）】

职责：拿到用户提问后，先"检索"（Retrieve），把检索结果拼进提示词，
      再交给大模型"生成"（Generate）基于资料的概括回答 —— 即 RAG 中的 R + AG。

运行时机：用户每次提问都会执行（高频、轻量、要求响应快）。
前置条件：vector_store.py 已经建好向量库，否则检索结果为空。
依赖方向：vector_store.py（写库） <-- rag_service.py（读库），单向依赖。
'''
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.vector_store import VectorStoreService

from utils.prompt_loader import load_rag_prompts


def print_prompt(prompt):
    # 调试观察点：原样透传 prompt，只把最终送给模型的内容打印出来
    # 用来确认 {input} / {context} 是否真的被填上（排查"检索到了却答不出"很有用）
    # 注意：Agent 正式调用时也会打印，联调结束建议从 chain 里摘掉
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagSummarizeService(object):  #将用户提问和根据用户提问检索到的参考资料一并交给模型，模型生成简洁准确的概括回答
    """RAG 问答服务：检索资料 -> 拼提示词 -> 让模型总结。"""

    def __init__(self):
        # 复用写库那边的向量库连接，只取检索器；这里不会触发任何入库动作
        self.chroma_vector_store_search = VectorStoreService()  #self.chroma_vector_store_search是为self.retriever服务的
        self.retriever = self.chroma_vector_store_search.get_retriever() #这个retriever是vector_store.py文件下VectorStoreService类的get_retriever()方法的实例，所以self.retriever写在类成员里
        # 提示词来自 prompts/rag_summarize.txt，含 {input} 和 {context} 两个占位符
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        # LangChain 管道写法：提示词 -> (调试打印) -> 大模型 -> 取出纯文本
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self,query: str)-> list[Document]:
        # 检索阶段：把提问向量化，在 Chroma 里找最相近的 k 个分片（k 由 chroma.yml 控制）
        return self.retriever.invoke(query)

    def rag_summarize(self,query: str)-> str:
        # ---------- R：检索 ----------
        context_docs = self.retriever_docs(query)

        # ---------- 拼上下文：多个分片编号后拼成一段文本，塞进 {context} ----------
        # 编号便于模型区分不同参考资料；元数据（来源/页码）留给模型判断依据
        context = " "
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"[参考资料{counter}]:参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # ---------- G：生成 ----------
        # 提问 + 参考资料一起交给模型，由 prompt 里的约束保证"只依据资料回答"
        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )

if __name__ == '__main__':
    # 独立运行本文件 = 单测"检索 + 生成"整条链路（前提：向量库中已有数据）
    rag = RagSummarizeService()

    print(rag.rag_summarize("进攻战术有哪些"))



