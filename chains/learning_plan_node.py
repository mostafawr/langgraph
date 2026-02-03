from pathlib import Path
from typing import Any, List
import json
import os

from dotenv import load_dotenv
from rich.console import Console

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.tools import TavilySearchResults
from langchain_groq import ChatGroq
from pydantic import BaseModel

# Default employees used when user input is missing or unparsable
DEFAULT_EMPLOYEES = [
    {"name": "Alice", "skill_gaps": ["NumPy", "Data Cleaning"]},
    {"name": "Bob", "skill_gaps": ["SQL Joins", "Data Modeling"]},
    {"name": "Charlie", "skill_gaps": ["Machine Learning", "Feature Engineering"]},
]

# Load env so API keys are available (e.g. TAVILY_API_KEY, MODAL_API_KEY if you add auth)
load_dotenv(Path(__file__).resolve().parent.parent / "1_introdction" / ".env")

console = Console()


class EmployeePlan(BaseModel):
    name: str
    learning_plan: List[str]


class LearningPlanOutput(BaseModel):
    employees: List[EmployeePlan]


def extract_json_string(content: Any) -> str:
    """Extract JSON string from plain text or markdown fences."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block["text"]
            elif isinstance(block, str):
                text += block
    else:
        text = str(content)

    text = text.strip()
    # Strip ``` fences
    if text.startswith("```"):
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    # Strip leading "json"
    if text.lower().startswith("json"):
        text = text[4:]
    return text.strip()


def build_employee_prompt(emp_list: List[dict]) -> str:
    txt = "Generate learning paths for these employees:\n"
    for emp in emp_list:
        txt += f"- {emp['name']}: {', '.join(emp['skill_gaps'])}\n"
    return txt


def _coerce_employees(obj: Any):
    """Normalize various JSON payload shapes into the employee list schema."""
    # Accept {"employees": [...]}
    if isinstance(obj, dict) and "employees" in obj:
        return _coerce_employees(obj.get("employees"))

    # Accept list of dicts with name + skills/gaps
    if isinstance(obj, list):
        normalized = []
        for item in obj:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            gaps = (
                item.get("skill_gaps")
                or item.get("skills")
                or item.get("gaps")
            )
            if not name or not isinstance(gaps, list):
                continue
            cleaned_gaps = [str(s).strip() for s in gaps if str(s).strip()]
            if cleaned_gaps:
                normalized.append(
                    {"name": str(name).strip(), "skill_gaps": cleaned_gaps}
                )
        if normalized:
            return normalized
    return None


def _parse_flexible_text(text: str):
    """
    Parse simple "Name: gap1, gap2" lines into the employee schema.
    Example input:
      Alice: NumPy, Data Cleaning
      Bob: SQL Joins, Data Modeling
    """
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    employees = []
    for line in lines:
        if ":" not in line:
            continue
        name, gaps_str = line.split(":", 1)
        gaps = [g.strip() for g in gaps_str.split(",") if g.strip()]
        if name.strip() and gaps:
            employees.append({"name": name.strip(), "skill_gaps": gaps})
    return employees or None


def parse_employees_from_state(state: List[BaseMessage]):
    """
    Try to extract an employee list from the first HumanMessage in the state.
    Supports JSON payloads or simple "Name: gap1, gap2" lines.
    Falls back to DEFAULT_EMPLOYEES if nothing parsable is found.
    """
    for msg in state:
        if not isinstance(msg, HumanMessage):
            continue
        try:
            # Try JSON first
            json_text = extract_json_string(msg.content)
            data = json.loads(json_text)
            parsed = _coerce_employees(data)
            if parsed:
                return parsed
        except Exception:
            pass

        # Try simple text format as a fallback
        parsed_text = _parse_flexible_text(msg.content)
        if parsed_text:
            return parsed_text
    return DEFAULT_EMPLOYEES


# ---------- TOOLS (Tavily search) ----------

search_tool = TavilySearchResults()
USE_TAVILY = os.getenv("USE_TAVILY", "1") == "1"
tools = [search_tool] if USE_TAVILY else []


# ---------- LLM (Groq) ----------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are an AI that generates structured, JSON-formatted learning paths for employees.

RULES:
1. The output MUST be valid JSON.
2. Do NOT include explanations, markdown, or commentary — ONLY JSON.
3. Follow this schema exactly:

{
  "employees": [
    {
      "name": "string",
      "learning_plan": ["string", "string", ...]
    }
  ]
}

4. Each learning plan must contain 4–7 actionable steps.
5. Each step may include direct links to recommended resources.

""",
)

def learning_plan_node(state: List[BaseMessage]) -> List[BaseMessage]:
    """
    Node callable for LangGraph: returns an AIMessage with JSON learning plans.

    """
    employees = parse_employees_from_state(state)
    user_prompt = build_employee_prompt(employees)

    result = agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
    raw = result["messages"][-1].content
    json_text = extract_json_string(raw)

    # Validate with Pydantic; fall back to error envelope if needed
    try:
        parsed = json.loads(json_text)
        validated = LearningPlanOutput(**parsed)
        json_text = validated.model_dump_json(indent=2)
    except Exception as e:
        json_text = json.dumps(
            {"employees": [], "error": f"Validation failed: {e}", "raw_response": raw}
        )

    return [AIMessage(content=json_text)]


if __name__ == "__main__":
    # Quick manual run
    msgs = learning_plan_node([])
    console.rule("[bold cyan]Learning Plan JSON")
    for msg in msgs:
        console.print(msg.content)
