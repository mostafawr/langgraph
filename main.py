#!/usr/bin/env python3
"""
Main execution file for the project analysis and team building workflow.
"""
import json
import os
import tkinter as tk
from tkinter import filedialog
from pypdf import PdfReader

from chains.final_workflow import app, FinalState

def select_pdf_file():
    """Opens a file dialog to select a PDF file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF files", "*.pdf")]
    )
    return file_path


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

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
        project_description = input("Enter the project description: ").strip()
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
