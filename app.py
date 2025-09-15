# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
import os
from typing import List, Dict, Any
from datasets import load_dataset
import pandas as pd

# LangChain imports
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain.schema import Document

# Gemini imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Config:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBZNWhMXa9SG0WDKbK5uhLc5ewxFmOyH_Y")
        self.model_name = "models/embedding-001"
        self.chat_model = "gemini-2.0-flash"
        self.chroma_db_path = "./chroma_db"
        self.chunk_size = 1250
        self.chunk_overlap = 100
        self.temperature = 0.0
        self.max_iterations = 30
        self.max_execution_time = 600


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


class MedicalAssistant:
    def __init__(self, config=None):
        self.config = config or Config()
        self.llm = None
        self.embedding_model = None
        self.vector_db = None
        self.agent_executors = {}  # Store multiple agents by session_id
        self.initialized = False

    async def initialize(self):
        """Initialize the medical assistant asynchronously"""
        try:
            logger.info("Initializing Medical Assistant...")

            # Initialize Gemini models
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.chat_model,
                google_api_key=self.config.gemini_api_key,
                temperature=self.config.temperature,
            )

            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model=self.config.model_name,
                google_api_key=self.config.gemini_api_key
            )

            # Load and process dataset
            await self._load_and_process_data()

            self.initialized = True
            logger.info("Medical Assistant initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Medical Assistant: {e}")
            raise

    async def _load_and_process_data(self):
        """Load and process the medical dataset"""
        try:
            logger.info("Loading medical dataset...")

            # Load dataset
            data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
            data = data.to_pandas()
            data = data.head(100)  # Limit for demo

            # Create documents
            df_loader = DataFrameLoader(data, page_content_column="Answer")
            documents = df_loader.load()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            texts = text_splitter.split_documents(documents)

            # Create vector store
            self.vector_db = Chroma.from_documents(
                documents=texts,
                embedding=self.embedding_model,
                persist_directory=self.config.chroma_db_path
            )

            logger.info("Data processing completed successfully")

        except Exception as e:
            logger.error(f"Error loading and processing data: {e}")
            raise

    def _create_agent_executor(self, session_id: str = "default"):
        """Create a new agent executor for a session"""
        try:
            logger.info(f"Creating medical agent for session {session_id}...")

            # Create retrieval QA chain
            qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_db.as_retriever(),
                return_source_documents=True
            )

            # Define tools with wrapper function to handle the dictionary output
            def medical_kb_wrapper(query: str) -> str:
                """Wrapper that extracts just the result from the QA chain output"""
                result_dict = qa.invoke({"query": query})
                return result_dict["result"]

            tools = [
                Tool(
                    name='Medical KB',
                    func=medical_kb_wrapper,
                    description=(
                        "Use this tool when answering medical knowledge queries to get "
                        "more information about the topic. Input should be a medical question."
                    )
                )
            ]

            # Create memory
            conversational_memory = ConversationBufferWindowMemory(
                memory_key='chat_history',
                k=4,
                return_messages=True
            )

            # Create agent
            prompt = hub.pull("hwchase17/react-chat")
            agent = create_react_agent(
                tools=tools,
                llm=self.llm,
                prompt=prompt,
            )

            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                memory=conversational_memory,
                max_iterations=self.config.max_iterations,
                max_execution_time=self.config.max_execution_time,
                handle_parsing_errors=True
            )

            self.agent_executors[session_id] = agent_executor
            logger.info(f"Agent created successfully for session {session_id}")

            return agent_executor

        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise

    async def process_message(self, message: str, session_id: str = "default") -> str:
        """Process a user message and return assistant response"""
        if not self.initialized:
            await self.initialize()

        try:
            # Get or create agent executor for this session
            if session_id not in self.agent_executors:
                self._create_agent_executor(session_id)

            agent_executor = self.agent_executors[session_id]

            # Process the message
            result = await asyncio.to_thread(
                agent_executor.invoke,
                {"input": message}
            )

            return result.get("output", "I'm sorry, I couldn't process your request.")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error while processing your message. Please try again."

    async def clear_memory(self, session_id: str = "default"):
        """Clear conversation memory for a specific session"""
        if session_id in self.agent_executors and hasattr(self.agent_executors[session_id], 'memory'):
            self.agent_executors[session_id].memory.clear()
            return True
        return False

    async def delete_session(self, session_id: str):
        """Delete a session and its memory"""
        if session_id in self.agent_executors:
            del self.agent_executors[session_id]
            return True
        return False


# Global instance
medical_assistant = MedicalAssistant()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat messages via HTTP POST"""
    try:
        response = await medical_assistant.process_message(request.message, request.session_id)
        return ChatResponse(response=response, session_id=request.session_id)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/")
async def root():
    return {"message": "Medical Assistant API is running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "initialized": medical_assistant.initialized,
        "active_sessions": len(medical_assistant.agent_executors)
    }


@app.post("/sessions/{session_id}/clear")
async def clear_session_memory(session_id: str):
    """Clear memory for a specific session"""
    success = await medical_assistant.clear_memory(session_id)
    if success:
        return {"message": f"Conversation memory cleared for session {session_id}"}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session completely"""
    success = await medical_assistant.delete_session(session_id)
    if success:
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.post("/sessions/clear-all")
async def clear_all_sessions():
    """Clear all sessions"""
    medical_assistant.agent_executors.clear()
    return {"message": "All sessions cleared successfully"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)