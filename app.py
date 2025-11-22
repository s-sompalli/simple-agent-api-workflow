import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from typing import Annotated
import os
import requests
from datetime import datetime

# Secure API key handling
if "ANTHROPIC_API_KEY" not in os.environ:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("Please set your Anthropic API key in .streamlit/secrets.toml")
        st.stop()
    os.environ["ANTHROPIC_API_KEY"] = api_key

# News API function
def search_news(query: str, max_results: int = 5) -> str:
    """
    Search for recent news articles about a topic.
    
    Args:
        query: The search term (e.g., "fortnite", "technology", "sports")
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Formatted string with news articles
    """
    try:

        news_api_key = st.secrets.get("NEWS_API_KEY", "")
        
        if not news_api_key:
            # Fallback to a simpler news aggregator or mock data
            return f"News API key not configured. Please add NEWS_API_KEY to your secrets.toml file. Search query was: {query}"
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": news_api_key,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": max_results
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] != "ok" or not data.get("articles"):
            return f"No news articles found for '{query}'"
        
        # Format the results
        results = [f"📰 News Results for '{query}':\n"]
        for i, article in enumerate(data["articles"][:max_results], 1):
            title = article.get("title", "No title")
            description = article.get("description", "No description")
            source = article.get("source", {}).get("name", "Unknown source")
            url = article.get("url", "")
            published = article.get("publishedAt", "")
            
            # Format date
            if published:
                try:
                    date_obj = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    published = date_obj.strftime("%B %d, %Y")
                except:
                    pass
            
            results.append(f"\n{i}. **{title}**")
            results.append(f"   Source: {source} | {published}")
            results.append(f"   {description}")
            if url:
                results.append(f"   Link: {url}")
        
        return "\n".join(results)
        
    except requests.RequestException as e:
        return f"Error fetching news: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def search_news_free(query: str, max_results: int = 5) -> str:
    """
    Search for news using a free RSS feed aggregator.
    This is a backup option if you don't have NewsAPI key.
    """
    try:
        # Using GNews API (no key required for basic usage)
        url = f"https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "lang": "en",
            "max": max_results,
            "apikey": "your-gnews-api-key"  # Get free key from gnews.io
        }
        
        # For demo purposes, return a formatted message
        return f"🔍 Searching for news about '{query}'...\n\nTo enable live news search:\n1. Get a free API key from https://newsapi.org/ or https://gnews.io/\n2. Add it to .streamlit/secrets.toml as NEWS_API_KEY"
        
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize Anthropic with tools
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.7
)

# Bind the tool to the LLM
tools = [search_news]
llm_with_tools = llm.bind_tools(tools)

# Define chatbot state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create Graph
graph_builder = StateGraph(State)

def chatbot(state: State):
    """Main chatbot node that decides whether to use tools"""
    langchain_messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict):
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        else:
            langchain_messages.append(msg)
    
    response = llm_with_tools.invoke(langchain_messages)
    return {"messages": [response]}

def should_continue(state: State):
    """Determine if we need to call tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise end
    return END

# Add nodes
graph_builder.add_node("chatbot", chatbot)

# Create tool node
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()

# Streamlit UI
st.title("🤖 Anthropic Chatbot with News Search")

st.sidebar.markdown("""
### Features
- 💬 Chat with Claude
- 📰 Search latest news
- 🔍 Automatic tool detection

### Example prompts:
- "What's the latest news about Fortnite?"
- "Find news about AI developments"
- "Show me recent articles on climate change"
""")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "processing" not in st.session_state:
    st.session_state.processing = False

# Display conversation
for message in st.session_state.history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message['content'])
    else:
        with st.chat_message("assistant"):
            st.write(message['content'])

# Input handling
user_input = st.chat_input("Type your message here...")

if user_input and not st.session_state.processing:
    st.session_state.processing = True
    
    # Add user message
    st.session_state.history.append({"role": "user", "content": user_input})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    try:
        # Stream response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for event in graph.stream({"messages": st.session_state.history}):
                for value in event.values():
                    if "messages" in value:
                        last_message = value["messages"][-1]
                        
                        # Handle different message types
                        if hasattr(last_message, "content") and last_message.content:
                            if isinstance(last_message.content, str):
                                full_response = last_message.content
                            elif isinstance(last_message.content, list):
                                # Handle content blocks
                                for block in last_message.content:
                                    if hasattr(block, "text"):
                                        full_response += block.text
                            
                            response_placeholder.markdown(full_response)
            
            if full_response:
                st.session_state.history.append({
                    "role": "assistant", 
                    "content": full_response
                })
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        st.session_state.history.append({
            "role": "assistant",
            "content": error_msg
        })
    finally:
        st.session_state.processing = False
    
    st.rerun()

# Add a clear button
if st.sidebar.button("Clear Conversation"):
    st.session_state.history = []
    st.rerun()
