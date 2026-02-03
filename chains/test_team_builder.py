
import json
import os
from team_builder_node import team_builder_node, TeamBuilderState

def test_team_builder():
    """
    Tests the team_builder_node with dummy data.
    """
    # --- 1. Create dummy project analysis file ---
    project_analysis_data = {
        "name": "Test Project",
        "description": "A test project for the team builder.",
        "tasks": [
            {
                "name": "Frontend Development",
                "description": "Develop the user interface.",
                "skills": ["React", "CSS"],
                "duration_days": 10,
                "start_days_from_kickoff": 0,
                "dependencies": []
            },
            {
                "name": "Backend Development",
                "description": "Develop the server-side logic.",
                "skills": ["Python", "Flask"],
                "duration_days": 15,
                "start_days_from_kickoff": 5,
                "dependencies": ["Frontend Development"]
            },
            {
                "name": "Database Management",
                "description": "Setup and manage the database.",
                "skills": ["SQL", "PostgreSQL"],
                "duration_days": 5,
                "start_days_from_kickoff": 10,
                "dependencies": ["Backend Development"]
            }
        ]
    }
    with open("project_analysis.json", "w") as f:
        json.dump(project_analysis_data, f, indent=2)

    # --- 2. Create dummy employee skills summary file ---
    if not os.path.exists("skill_results"):
        os.makedirs("skill_results")
        
    employee_skills_data = [
        {
            "filename": "employee1.json",
            "summary": {
                "Web Development": ["React", "JavaScript"],
                "Styling": ["CSS"]
            }
        },
        {
            "filename": "employee2.json",
            "summary": {
                "Backend Development": ["Python", "Django"],
                "Databases": ["MySQL"]
            }
        },
        {
            "filename": "employee3.json",
            "summary": {
                "Databases": ["PostgreSQL", "SQL"],
                "Cloud": ["AWS"]
            }
        }
    ]
    with open("skill_results/all_skills_summary.json", "w") as f:
        json.dump(employee_skills_data, f, indent=2)

    # --- 3. Run the team builder node ---
    state = TeamBuilderState(
        num_employees=3,
        project_analysis={},  # Not used directly, loaded from file
        employee_skills=[],  # Not used directly, loaded from file
        team={}
    )
    result = team_builder_node(state)

    # --- 4. Print the result ---
    print(json.dumps(result, indent=2))

    # --- 5. Clean up dummy files ---
    os.remove("project_analysis.json")
    os.remove("skill_results/all_skills_summary.json")
    os.rmdir("skill_results")


if __name__ == "__main__":
    test_team_builder()
