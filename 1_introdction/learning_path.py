from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
from rich.console import Console
import os
import json
from typing import List
from pydantic import BaseModel

# ========================================
# 1. Load environment + setup rich console
# ========================================
load_dotenv()
console = Console()


# ========================================
# 2. Pydantic Models for Validation
# ========================================

class EmployeePlan(BaseModel):
    name: str
    learning_plan: List[str]

class LearningPlanOutput(BaseModel):
    employees: List[EmployeePlan]


# ========================================
# 3. Static employees (as requested)
# ========================================
employees = [
    {"name": "Alice", "skill_gaps": ["NumPy", "Data Cleaning"]},
    {"name": "Bob", "skill_gaps": ["SQL Joins", "Data Modeling"]},
    {"name": "Charlie", "skill_gaps": ["Machine Learning", "Feature Engineering"]},
]


# ========================================
# 4. Extraction Helpers
# ========================================

def extract_json_string(content):
    """Extracts JSON string from:
    - plain string
    - LC message blocks
    - markdown ```json fenced code
    """
    # Case 1: already str
    if isinstance(content, str):
        text = content
    # Case 2: LC v0.3 content blocks
    elif isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block["text"]
            elif isinstance(block, str):
                text += block
    else:
        text = str(content)

    # Remove ```json fences
    text = text.strip()
    if text.startswith("```"):
        # remove opening fence
        text = text.split("```", 1)[1]
        # remove closing fence if exists
        if "```" in text:
            text = text.split("```", 1)[0]

    # Clean leftover "json" language tags
    text = text.replace("json", "", 1).strip()

    return text.strip()


# ========================================
# 5. Build the prompt from static employees
# ========================================
def build_employee_prompt(employees):
    txt = "Generate learning paths for these employees:\n"
    for emp in employees:
        txt += f"- {emp['name']}: {', '.join(emp['skill_gaps'])}\n"
    return txt


# ========================================
# 6. LLM + Agent Setup (Gemini + Tool)
# ========================================
searchtool = TavilySearchResults()
tools = [searchtool]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

system_prompt = """
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
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


# ========================================
# 7. Invoke the agent
# ========================================
user_prompt = build_employee_prompt(employees)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": user_prompt}
    ]
})

raw_output = response["messages"][-1].content


console.rule("[bold cyan]RAW MODEL OUTPUT")
console.print(raw_output)


# ========================================
# 8. Extract JSON String + Validate
# ========================================
try:
    json_string = extract_json_string(raw_output)
    
    parsed = json.loads(json_string)  # <-- FIXED: now receiving clean JSON
    validated = LearningPlanOutput(**parsed)

    console.rule("[bold green]VALIDATED LEARNING PLANS")
    console.print(validated.model_dump_json(indent=2))

except Exception as e:
    console.rule("[bold red]VALIDATION ERROR")
    console.print(e)
    console.print("\nJSON STRING WAS:\n", json_string)
