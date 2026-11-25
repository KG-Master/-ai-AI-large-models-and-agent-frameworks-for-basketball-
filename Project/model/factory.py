from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from utils.config_handler import rag_conf
#Ollama本地模型导包
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
#阿里云通义系列模型导包
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
'''
模型用的比较多或者有切换需求时(2、3个模型甚至更多)，一般弄个像这样的工厂文件【factory.py】
实现工厂文件一般都需要实现一个生成器，生成器用于生成想要的对象
'''
'''
**工厂方法模式（Factory Method）**，是 RAG 项目里很标准的写法，核心目的：**把【模型实例的创建逻辑】和【业务调用逻辑拆开、解耦】**。
## 为什么要这么写:
### 1. 切换模型 / 服务商非常方便，业务代码不用大改
现在你用的是阿里通义 DashScope。以后想换成 OpenAI、Ollama、本地模型：
- 只新增一个工厂类（比如 `OpenAIChatFactory`），实现 `generator`
- 上层 RAG、chain、检索的代码**一行都不用改**
> 
> 如果不写工厂，到处直接写 `ChatTongyi(xxx)`，以后换模型要全局搜索替换，很容易漏、改错。
'''


#先定义一个抽象类
class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]: #当前抽象类定义的抽象方法返回对象的类型要么是Embeddings实例，要么是BaseChatModel实例
        pass  #抽象类的抽象方法不定义具体的行为，定义一个生成器，返回值类型是可选的，要么是Embeddings嵌入向量模型，要么是BaseChatModel基础聊天模型
    #抽象类定义好了后，下面要实现BaseModelFactory抽象类的子类实现



class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOllama(
            # 模型名，从 rag.yml 的 ollama_chat_model 读取
            model=rag_conf["ollama_chat_model"],
            # 服务地址；yml 里没写时用第二个参数作兜底值
            base_url=rag_conf.get("ollama_base_url", "http://127.0.0.1:11434"),
            # 温度 0~1：越大越发散，0.3 偏稳定，适合知识库问答
            temperature=0.3,
            # 空闲后模型在显存中保留多久；"30m"=30 分钟，"0"=用完立刻卸载
            keep_alive=rag_conf.get("keep_alive", "30m"),
            # 上下文长度；yml 读出来是 int，这里再转一次，防止写成字符串导致报错
            num_ctx=int(rag_conf.get("num_ctx", 8192)),
        )

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return OllamaEmbeddings(
            model=rag_conf["ollama_embedding_model"],
            base_url=rag_conf.get("ollama_base_url", "http://127.0.0.1:11434"),
        )


# class ChatModelFactory(BaseModelFactory):
#     '''
#     BaseModelFactory抽象类定义了@abstractmethod的generator的抽象方法，
#     ChatModelFactory继承了BaseModelFactory，BaseModelFactory是ChatModelFactory的父类
#     子类必须具体实现父类所定义的抽象方法
#     '''
#     def generator(self)-> Optional[Embeddings | BaseChatModel]:
#         return ChatTongyi(model=rag_conf["chat_model_name"])
#
#
#
# class EmbeddingsFactory(BaseModelFactory):
#     def generator(self)-> Optional[Embeddings | BaseChatModel]:
#         return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])

#快捷获取
chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()