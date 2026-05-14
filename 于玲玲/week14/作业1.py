import os
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PyMuPDFLoader, PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


class FinancialRAGSystem:
    def __init__(self, es_url: str,index_name: str):
        """
        初始化 RAG 系统核心组件
        """
        self.index_name = index_name

        # 1. 初始化 Elasticsearch 客户端
        self.es_client = Elasticsearch(
            es_url
        )

        # 2. 初始化大模型
        self.llm = ChatOpenAI(
            model="qwen-flash",  # 模型的代号
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-57fc36ecd653492484a132b3218dac57"
        )

        # 3. 初始化中文向量模型 (BGE)
        self.embedding_model = HuggingFaceEmbeddings(model_name="/Users/yull/AI/model/bge-small/")

        # 4. 初始化重排序模型 (Cross-Encoder)
        self.reranker = CrossEncoder('/Users/yull/AI/model/bge-reranker-base')

    def create_index(self):
        """
        在 Elasticsearch 中创建带有向量映射的索引
        """
        # 注意：dims=1024 必须与 bge 的向量维度保持一致
        index_mapping = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "text_content": {"type": "text"},
                    "vector_field": {"type": "dense_vector", "dims": len(self.embedding_model.embed_query("text")), "index": True, "similarity": "cosine"},
                    "tenant_id": {"type": "keyword"},
                    "source": {"type": "keyword"}
                }
            }
        }

        # if self.es_client.indices.exists(index=self.index_name):
        #     print(f"检测到旧索引 '{self.index_name}'，正在删除...")
        #     self.es_client.indices.delete(index=self.index_name)
        #     print(f"旧索引删除成功！")
        # else:
        #     print(f"索引 '{self.index_name}' 不存在，准备直接创建。")
        if self.es_client.indices.exists(index=self.index_name):

            print(f"索引 '{self.index_name}' 已存在，将直接复用。")
        else:
            self.es_client.indices.create(index=self.index_name, body=index_mapping)
            print(f"索引 '{self.index_name}' 创建成功！")

    def build_knowledge_base(self, file_dir: str, tenant_id: str):
        """
        解析本地文档，分块并向量化存入 ES
        """
        print(f"开始为租户 [{tenant_id}] 构建知识库...")

        # 1. 加载文档 (支持 PDF)
        loader = DirectoryLoader(file_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        # loader = PDFPlumberLoader("data/lk/汽车知识手册-领克.pdf")
        documents = loader.load()

        # 2. 文本分块 (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)

        # 3. 注入多租户元数据
        for doc in split_docs:
            doc.metadata.update({"tenant_id": tenant_id, "source": doc.metadata.get("source")})

        # 4. 存入 Elasticsearch
        ElasticsearchStore.from_documents(
            documents=split_docs,
            embedding=self.embedding_model,
            es_connection=self.es_client,
            index_name=self.index_name,
            vector_query_field="vector_field",
            query_field="text_content"
        )
        print(f"知识库构建完成，共写入 {len(split_docs)} 个文本块。")

    def _retrieve_with_rerank(self, query: str, tenant_id: str, top_k: int = 3, fetch_k: int = 10) -> List[Document]:
        """
        内部方法：执行带多租户隔离的混合检索与重排序
        """
        vectorstore = ElasticsearchStore(
            index_name=self.index_name,
            embedding=self.embedding_model,
            es_connection=self.es_client,
            query_field="text_content",
            vector_query_field="vector_field"
        )

        # 强制多租户过滤
        filter_dict = {"term": {"metadata.tenant_id": tenant_id}}

        # 第一阶段：混合检索召回
        retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k, "filter": filter_dict})
        initial_docs = retriever.invoke(query)

        if not initial_docs:
            return []

        # 第二阶段：Cross-Encoder 重排序
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)

        # 按分数排序并截取 Top K
        ranked_docs = [doc for _, doc in sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)]
        return ranked_docs[:top_k]

    def ask(self, query: str, tenant_id: str) -> Dict[str, Any]:
        """
        对外提供的 RAG 问答接口
        """
        # 1. 获取高质量上下文
        context_docs = self._retrieve_with_rerank(query, tenant_id)

        if not context_docs:
            return {"answer": "抱歉，在当前租户的知识库中未找到相关信息。", "citations": []}

        # 2. 构造防幻觉 Prompt
        context_text = "\n\n".join([
            f"[来源: {os.path.basename(doc.metadata.get('source', '未知文件'))}] {doc.page_content}"
            for doc in context_docs
        ])

        prompt_template = f"""你是一名汽车专家。请严格基于以下参考资料回答用户的问题。
【约束条件】
1. 答案必须完全基于提供的参考资料，严禁编造。
2. 如果资料中没有相关信息，请直接说明未找到。
3. 回答时，请在相关句子的末尾用方括号标注引用的来源文件名。

【参考资料】
{context_text}

【用户问题】
{query}

【你的回答】
"""
        # 3. 调用本地 Qwen 生成回答
        response = self.llm.invoke(prompt_template)

        # 4. 提取引用来源
        sources = list(set([os.path.basename(doc.metadata.get('source', '未知文件')) for doc in context_docs]))

        return {"answer": response.content, "citations": sources}


# ================== 运行测试 ==================
if __name__ == "__main__":
    # 1. 实例化系统 (请替换为你真实的 ES 密码和索引名)
    rag = FinancialRAGSystem(
        es_url="http://localhost:9200",
        index_name="lingke"
    )

    # 2. 创建索引 (首次运行执行一次)
    rag.create_index()

    # 3. 构建知识库
    rag.build_knowledge_base(file_dir="./data/lk", tenant_id="lingke")

    # # 4. 测试问答
    result = rag.ask(query="座椅如何调节？", tenant_id="lingke")
    print("回答：")
    print(result["answer"])
    print("引用来源：", result["citations"])