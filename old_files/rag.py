from langchain.document_loaders import WebBaseLoader

#load web chain
loader_url = 'https://lilianweng.github.io/posts/2023-06-23-agent/'
loader = WebBaseLoader(loader_url)
data = loader.load()

#split into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, 
                                               chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

#embed and store
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import OllamaEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits,
                                    embedding=GPT4AllEmbeddings())

#Retreive
question = "How can Task Decomposition be done?"
docs = vectorstore.similarity_search(question)
len(docs)

#Rag prompt
from langchain import hub
QA_CHAIN_PROMPT = hub.pull('rlm/rag-prompt-llama')

#LLM
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model='llama2',
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

#QA chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={'prompt': QA_CHAIN_PROMPT},
)

question = 'What are the various approaches to Task Decomposition for AI Agents?'
result = qa_chain({'query': question})

#get logging for tokens
from langchain.schema import LLMResult
from langchain.callbacks.base import BaseCallbackHandler

class GenerationStatisticsCallback(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(response.generations[0][0].generation_info)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler(),
                                    GenerationStatisticsCallback()])

llm = Ollama(base_url='http://localhost:11434',
             model='llama2',
             verbose=True,
             callback_manager=callback_manager)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={'prompt': QA_CHAIN_PROMPT},
)

question = 'What are the approaches to Task Decomposition?'
result = qa_chain({'query': question})

#use the hub for prompt managment

