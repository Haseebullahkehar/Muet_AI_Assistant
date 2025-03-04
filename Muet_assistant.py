import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferWindowMemory

# Access API keys from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

if not openai_api_key:
    st.error(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

if not tavily_api_key:
    st.error(
        "Tavily API key not found. Please set the TAVILY_API_KEY environment variable.")
    st.stop()

# Initialize memory
memory = ConversationBufferWindowMemory(
    k=5, memory_key="chat_history", return_messages=True)

# Load and process document
loader = PyPDFLoader("data/Muet_Prospectus-23.pdf")
loaded_document = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=5)
chunks = text_splitter.split_documents(loaded_document)

# Generate embeddings using OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Define prompt template
template = """You are a helpful assistant for Mehran University of Engineering and Technology (MUET). Use the following context to answer the question. If you don't know the answer, use real-time search to find relevant information.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Initialize OpenAI LLM and tools
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
search = TavilySearchResults(api_key=tavily_api_key, max_results=1)
tools = [search]
agent_executor = create_react_agent(llm, tools)

# Streamlit UI
st.set_page_config(
    page_title="MUET AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with profile links and project description
with st.sidebar:
    st.image("data/Muet-logo.png", width=150)
    st.markdown("""
        # Welcome to MUET AI Assistant
        Your virtual assistant for all things related to Mehran University of Engineering and Technology (MUET). 
        Powered by advanced AI, this assistant is here to answer your queries about MUET.
    """)
    st.markdown("---")
    st.markdown("### Developer: Haseebullah Kehar")
    st.markdown("""
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/haseebullah-kehar-203021243/)
        [![GitHub](https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/Haseebullahkehar)
    """)
    st.markdown("---")

# Main chat interface
st.title("MUET AI Assistant")
st.write("Ask me anything about Mehran University of Engineering and Technology!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Enter your query:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({
                "messages": [HumanMessage(content=user_input)],
                "chat_history": memory.load_memory_variables({})["chat_history"]
            })
            answer = response["messages"][-1].content if "messages" in response else "No response received."
            st.markdown(answer)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
