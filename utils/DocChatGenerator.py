# Copyright (c) Microsoft. All rights reserved.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader

from . import InferenceGenerator
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
os.environ["OPENAI_API_KEY"] = "sk-"

# The model used to get the tokenizer can be a little arbitrary
# since the tokenizers are common within the same model type


class DocChatGenerator(InferenceGenerator.InferenceGenerator):
    model = None
    embeddings = None
    llm = None

    def __init__(self, model_name):
        super().__init__(model_name)
        # 初始化 openai embeddings
        if not DocChatGenerator.embeddings:
            model_kwargs = {'device': 'mps'}
            DocChatGenerator.embeddings = HuggingFaceEmbeddings(model_name='GanymedeNil/text2vec-large-chinese', model_kwargs=model_kwargs, cache_folder="./")
        self.embeddings = DocChatGenerator.embeddings

        if not DocChatGenerator.llm:
            DocChatGenerator.llm = HuggingFaceHub(repo_id="ClueAI/ChatYuan-large-v2", model_kwargs={"temperature": 0.8})
        self.llm = DocChatGenerator.llm

    def perform_inference(self, prompt, context, max_tokens):
        loader = UnstructuredFileLoader("./data.txt")
        # 将数据转成 document
        documents = loader.load()
        # 初始化文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50
        )
        # 分割 youtube documents
        documents = text_splitter.split_documents(documents)

        # 将数据存入向量存储
        # docsearch = Chroma.from_documents(documents, self.embeddings, persist_directory="./vector_store")
        # docsearch.persist()

        # 加载数据
        docsearch = Chroma(persist_directory="./vector_store", embedding_function=self.embeddings)

        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=docsearch.as_retriever(),  return_source_documents=True)
        # 进行问答
        result = qa({"query": prompt})
        print(result)

        return (
            result["result"],
            "",
            0,
        )
