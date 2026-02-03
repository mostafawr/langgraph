from pathlib import Path
from typing import List
import os

from dotenv import load_dotenv
from rich.console import Console

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, MessageGraph
from langchain_groq import ChatGroq

# Allow running as a script path by setting the package and sys.path.
if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "chains"

# Load environment variables (API keys, etc.) before importing chains
load_dotenv(Path(__file__).resolve().parent.parent / "1_introdction" / ".env")

from .learning_plan_node import learning_plan_node

console = Console()


def _clean_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Drop any empty-content messages to avoid API errors."""
    return [
        m
        for m in messages
        if getattr(m, "content", "") and str(m.content).strip()
    ]


graph = MessageGraph()

reflect = "reflect"
plan = "learning_plan"

# LLM for reflection (Groq)
reflection_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict reviewer. Given the last JSON learning plan, produce a concise critique only. "
            "List the gaps, missing links, unclear steps, or schema errors. "
            "Do NOT rewrite the plan or return JSON—just short bullet-style issues.",
        ),
        MessagesPlaceholder("messages"),
    ]
)
reflection_chain = reflection_prompt | reflection_llm


def plan_node(state: List[BaseMessage]) -> List[BaseMessage]:
    # Generates a learning plan (or improved plan) using the Tavily-backed node
    return learning_plan_node(state)


def reflect_node(state: List[BaseMessage]) -> List[BaseMessage]:
    # Only send the latest AI message to the reviewer to avoid confusion
    messages = _clean_messages([state[-1]]) if state else []
    if not messages:
        return []
    response = reflection_chain.invoke({"messages": messages})
    if not response.content or not str(response.content).strip():
        return []
    return [HumanMessage(response.content)]


graph.add_node(plan, plan_node)
graph.set_entry_point(plan)


def should_continue(state: List[BaseMessage]):
    # Stop after the first plan response (human -> plan AI)
    if len(state) >= 2:
        return END
    return plan


graph.add_conditional_edges(
    plan,
    should_continue,
    {END: END},
)

app = graph.compile()

if __name__ == "__main__":
    print(app.get_graph().draw_mermaid())
    app.get_graph().print_ascii()

    # Run a demo turn and pretty-print the conversation.
    demo_employees = """Alice: NumPy, Data Cleaning
Bob: SQL Joins, Data Modeling
Charlie: Machine Learning, Feature Engineering"""
    result = app.invoke([HumanMessage(content=demo_employees)])
    console.rule("[bold cyan]Conversation[/]")
    for msg in result:
        role = msg.__class__.__name__.replace("Message", "")
        console.print(f"[bold]{role}: [/]{msg.content}")
