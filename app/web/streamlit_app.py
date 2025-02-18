from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime 
import torch

import streamlit as st
from dotenv import load_dotenv
from supabase import Client
import logging

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart
)

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.deepseek_adapter import DeepSeekAdapter
from services.rag_service import RAGService
from pydantic import BaseModel, ConfigDict

torch.classes.__path__ = [] # @see: https://github.com/VikParuchuri/marker/issues/442 

# Load environment variables
load_dotenv()

# Initialize clients
supabase = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

model = DeepSeekAdapter(
    model_name=os.getenv("MODEL_NAME", "deepseek-r1:1.5b"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-es"),
    base_url="http://ollama:11434"
)

class ChatMessage(TypedDict):
    role: Literal['user', 'assistant', 'system']
    content: str
    timestamp: str

class PydanticAIDeps(BaseModel):
    supabase: Client
    model: DeepSeekAdapter
    rag_service: RAGService
    model_config = ConfigDict(arbitrary_types_allowed=True)

def display_message_part(part: ModelMessage) -> None:
    """Display a message part in the Streamlit UI."""
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)
    elif part.part_kind == 'tool-call':
        with st.chat_message("assistant"):
            st.info(f"Using tool: {part.tool_name}")
    elif part.part_kind == 'tool-return':
        with st.chat_message("assistant"):
            st.success(f"Tool result: {part.content[:100]}...")

async def run_agent_with_streaming(user_input: str, deps: PydanticAIDeps) -> None:
    """Run the RAG agent with streaming responses."""
    message_placeholder = st.empty()
    partial_response = ""

    try:
        # Prepare the context using RAG
        similar_chunks = await deps.rag_service.get_similar_chunks(user_input)
        
        if not similar_chunks:
            st.warning("No relevant documentation found. Please try a different question.")
            return
            
        context = "\n\n".join(chunk['content'] for chunk in similar_chunks)

    except Exception as e:
        st.error(f"Error in run_agent_with_streaming: {str(e)}")
        st.error(f"An error occurred while processing your request. Please try again later.")

    # Create system prompt with context
    system_message = SystemPromptPart(
        content=f"""You are a Web Craweler AI expert. Use the following documentation to answer questions:
        
        {context}
        
        If you can't find the information in the provided context, say so."""
    )

    # Create user message
    user_message = UserPromptPart(content=user_input)

    # Create request
    request = ModelRequest(parts=[system_message, user_message])
    
    try:
        async for chunk in deps.model.generate_text(
            prompt=user_input,
            system_prompt=system_message.content,
            stream=True
        ):
            partial_response += chunk
            message_placeholder.markdown(partial_response + "▌")

        # Create final response
        response = ModelResponse(parts=[TextPart(content=partial_response)])
        
        # Add to session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        st.session_state.messages.extend([request, response])

        # Show sources
        with st.expander("View Sources", expanded=False):
            for chunk in similar_chunks:
                st.markdown(f"### {chunk['title']}")
                st.markdown(chunk['content'])
                st.markdown("---")

    except Exception as e:
        st.error(f"Error: {str(e)}")

def validate_environment():
    required_vars = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "deepseek-r1:1.5b")
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


def main():
    try:
        validate_environment()
        st.title("Web Crawler AI Documentation Expert")
        st.write("Ask any question!")

        # Initialize services
        rag_service = RAGService(supabase, model)
        deps = PydanticAIDeps(
            supabase=supabase,
            model=model,
            rag_service=rag_service
        )

        # Initialize messages in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            if isinstance(msg, (ModelRequest, ModelResponse)):
                for part in msg.parts:
                    display_message_part(part)

        # Chat input
        if user_input := st.chat_input("Ask me anything..."):
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                asyncio.run(run_agent_with_streaming(user_input, deps))

    except Exception as e:
            st.error(f"Application configuration error: {str(e)}")
            return

if __name__ == "__main__":
    main()

