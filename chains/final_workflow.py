#!/usr/bin/env python3
"""
The final, integrated workflow for the project analysis and team building process.
"""
import json
from typing import List, Dict, Any, TypedDict
import os

from langgraph.graph import StateGraph, END

from . import extractor_with_langgraph
from .project_analyzer_node import project_analyzer_node, HumanMessage
from .team_builder_node import team_builder_node
from .project_analyzer_node import select_pdf_file, extract_text_from_pdf

# -----------------------------
# 1. OVERALL GRAPH STATE
# -----------------------------

class FinalState(TypedDict):
    project_description: str
    project_skills: List[str]
    num_employees: int


# -----------------------------
# 2. WRAPPER NODES
# -----------------------------

def run_skill_extractor(state: FinalState) -> None:
    """Runs the skill extractor graph."""
    print("--- Running Skill Extractor ---")
    extractor_with_langgraph.main()


def run_project_analyzer(state: FinalState) -> None:
    """Runs the project analyzer node."""
    print("--- Running Project Analyzer ---")
    analyzer_state = [
        HumanMessage(
            content=json.dumps(
                {
                    "skills": state["project_skills"],
                    "description": state["project_description"],
                }
            )
        )
    ]
    project_analyzer_node(analyzer_state)


def run_team_builder(state: FinalState) -> None:
    """Prepares the state and runs the team builder node."""
    print("--- Running Team Builder ---")
    builder_state = {
        "num_employees": state["num_employees"],
        # The other fields are now loaded from files within the node
        "project_analysis": {},
        "employee_skills": [],
        "team": {}
    }
    team_builder_node(builder_state)


# -----------------------------
# 3. BUILD THE FINAL GRAPH
# -----------------------------

workflow = StateGraph(FinalState)

workflow.add_node("skill_extractor", run_skill_extractor)
workflow.add_node("project_analyzer", run_project_analyzer)
workflow.add_node("team_builder", run_team_builder)

workflow.set_entry_point("skill_extractor")
workflow.add_edge("skill_extractor", "project_analyzer")
workflow.add_edge("project_analyzer", "team_builder")
workflow.add_edge("team_builder", END)

app = workflow.compile()

def main():
    """
    Main function to run the final workflow.
    """
    # --- Get Initial State ---
    choice = input("Choose input type (description/pdf): ").lower()

    if choice == "pdf":
        file_path = select_pdf_file()
        if not file_path:
            print("No file selected. Exiting.")
            return
        project_description = extract_text_from_pdf(file_path)
    else:
        print("Enter the project description (type 'EOF' on a new line when you're done):")
        lines = []
        while True:
            line = input()
            if line == "EOF":
                break
            lines.append(line)
        project_description = "\n".join(lines)

    project_skills_str = input("Enter initial skills (comma-separated): ")
    project_skills = [s.strip() for s in project_skills_str.split(",") if s.strip()]

    num_employees_str = input("Enter the desired team size: ")
    num_employees = int(num_employees_str) if num_employees_str.isdigit() else 22

    initial_state = {
        "project_description": project_description,
        "project_skills": project_skills,
        "num_employees": num_employees,
    }

    # --- Run the Workflow ---
    final_output = app.invoke(initial_state)

    print("\n--- Final Workflow Output ---")
    print(final_output)

if __name__ == "__main__":
    main()
