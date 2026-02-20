#!/usr/bin/env python3
"""
The final, integrated workflow for the project analysis and team building process.
"""
import sys
from typing import Dict, TypedDict

from langgraph.graph import StateGraph, END

try:
    from chains import extractor_with_langgraph
    from chains.common_imports import json, List, Any, os
    from chains.project_analyzer_node import project_analyzer_node, HumanMessage, select_pdf_file, extract_text_from_pdf
    from chains.team_builder_node import team_builder_node
except ImportError as e:
    print("\n" + "!"*60)
    print("❌ IMPORT ERROR: Could not load required modules.")
    print(f"Error details: {e}")
    print("-" * 60)
    print(f"You are currently running Python from:\n👉 {sys.executable}")
    print("\nMake sure you install packages to THIS environment using:")
    print(f"   {sys.executable} -m pip install langchain-groq langgraph pypdf")
    print("!"*60 + "\n")
    sys.exit(1)

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


# -----------------------------
# 4. MAIN EXECUTION
# -----------------------------

def main():
    """
    Main function to run the final workflow.
    """
    # --- Initial Inputs ---
    project_description = ""
    project_skills = []
    
    choice = input("Analyze project from 'description' or 'pdf'? ").lower().strip()

    if choice == "pdf":
        pdf_path = select_pdf_file()
        if pdf_path and os.path.exists(pdf_path):
            project_description = extract_text_from_pdf(pdf_path)
            skills_str = input("Enter initial skills (comma-separated), if any: ").strip()
            if skills_str:
                project_skills = [s.strip() for s in skills_str.split(",")]
        else:
            print("No PDF file selected or file not found.")
            return
    elif choice == "description":
        print("Enter the project description (type 'EOF' on a new line when you're done):")
        lines = []
        while True:
            line = input()
            if line == "EOF":
                break
            lines.append(line)
        project_description = "\n".join(lines)
        skills_str = input("Enter initial skills (comma-separated), if any: ").strip()
        if skills_str:
            project_skills = [s.strip() for s in skills_str.split(",")]
    else:
        print("Invalid choice. Please enter 'description' or 'pdf'.")
        return

    num_employees_str = input("Enter the number of employees for the team: ").strip()
    try:
        num_employees = int(num_employees_str)
    except ValueError:
        print("Invalid number of employees. Please enter an integer.")
        return

    initial_state: FinalState = {
        "project_description": project_description,
        "project_skills": project_skills,
        "num_employees": num_employees,
    }

    # --- Run the Workflow ---
    app.invoke(initial_state)

    # --- Read the final output ---
    try:
        with open("team_composition.json", "r", encoding="utf-8") as f:
            final_team = json.load(f)
        
        print("\n" + "="*60)
        print("           FINAL TEAM COMPOSITION           ")
        print("="*60)
        print(json.dumps(final_team, indent=2))
        print("="*60 + "\n")

    except FileNotFoundError:
        print("\n" + "="*60)
        print("Could not find team_composition.json. An error likely occurred during the workflow.")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
