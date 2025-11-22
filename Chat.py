import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

# Load API key from secrets
API_KEY = st.secrets["anthropic"]["api_key"]

# Initialize Anthropic
llm = ChatAnthropic(api_key=API_KEY, model="claude-3-5-sonnet-20240620")

# Define chatbot state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create Graph
graph_builder = StateGraph(State)

def chatbot(state: State):
    langchain_messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict):
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        else:
            langchain_messages.append(msg)
    response = llm.invoke(langchain_messages)
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Streamlit UI
st.title("Anthropic Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    for event in graph.stream({"messages": st.session_state.history}):
        for value in event.values():
            assistant_message = value["messages"][-1].content
            st.session_state.history.append({"role": "assistant", "content": assistant_message})

# Display conversation
for message in st.session_state.history:
    if message["role"] == "user":
        st.write(f"🧑: {message['content']}")
    else:
        st.write(f"🤖: {message['content']}")
