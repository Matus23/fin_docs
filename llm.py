import os
import yaml
import json
import litellm
# import ollama
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_community.document_loaders import PDFPlumberLoader
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, WikipediaTools


# Load and preprocess PDF financial statement
def load_pdf_text(path: str):
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def load_document(path: str, type: str = 'lumber'):
    assert type in ['lumber', 'pypdf'], 'Invalid load type'

    if type == 'lumber':
        loader = PDFPlumberLoader(path)
        docs = loader.load()
    elif type == 'pypdf':
        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            docs = [page.extract_text() for page in reader.pages if page.extract_text()]
            
    return docs



# Convert text into vector store
def create_vector_store(documents, model: str, load_type: str='lumber'):
    assert load_type in ['lumber', 'pypdf'], 'Invalid load type'

    if load_type == 'pypdf':
        text = "\n".join([page for page in documents])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

    embeddings = OllamaEmbeddings(model=model)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


if __name__ == '__main__':
    with open('config.json', 'r') as config_f:
        config = json.load(config_f)
        config = config[config['mode']]

    # Load Fidelity's latest annual report
    docs = load_document(path=config['file'], type=config['load_type'])
    print(f"Number of document chunks: {len(docs)}")
    vector_store = create_vector_store(docs, model=config['model'], load_type=config['load_type'])

    # Set up RAG pipeline
    retriever = vector_store.as_retriever()
    llm = OllamaLLM(model=config['model'])
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # for query in config['queries']:
    #     response = qa_chain.invoke(query)
    #     print(f"Query: {response['query']}\n Result: {response['result']}\n\n")

    ############################################################
    #    Agent expansion
    ############################################################
    with open('agents_config.yaml', "r") as file:
        ag_config = yaml.safe_load(file)

    search_tool = SerperDevTool()
    wiki_tool = WikipediaTools()

    agents = {}
    for agent_data in ag_config["agents"]:
        agent = Agent(
            role=agent_data["role"],
            goal=agent_data["goal"],
            backstory=agent_data["backstory"],
            llm=litellm.completion(model=f'ollama/{config['model']}', provider="ollama"),
            verbose=True,
            tools=[search_tool, wiki_tool]
        )
        agents[agent_data["role"]] = agent

    tasks = []
    for task_data in ag_config["tasks"]:
        task = Task(
            description=task_data["description"],
            agent=agents[task_data["agent"]],
            expected_output=task_data["expected_output"]
        )
        tasks.append(task)

    fin_crew = Crew(agents=list(agents.values()), tasks=tasks)

    for query in config['queries']:
        response = fin_crew.kickoff()
        print(f"Query: {query}\n Result: {response}\n\n")


