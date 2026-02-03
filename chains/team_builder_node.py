"""
Team Builder node for LangGraph workflows.
"""

from .common_imports import (
    BaseModel,
    List,
    Dict,
    Any,
    Field,
    TypedDict,
    json,
    RootModel,
    Tuple,
)
from .project_analyzer_node import ProjectTask, ProjectAnalysisOutput

import numpy as np
from .skills_graph import create_skills_graph, get_skill_relatives

# --- Initialize Skills Graph ---
SKILLS_GRAPH = create_skills_graph()

# -----------------------------
# 1. DATA SCHEMAS (Pydantic)
# -----------------------------

class EmployeeSkill(BaseModel):
    """Represents the skills of a single employee."""
    filename: str
    summary: Dict[str, List[str]]

class AllEmployeeSkills(RootModel[List[EmployeeSkill]]):
    """Represents a list of all employee skills."""
    pass

class EmployeeAssignment(BaseModel):
    """A single task assigned to an employee."""
    task_name: str
    start_day: int
    end_day: int
    skills_match: List[str]
    missing_skills: List[str]

class TeamMember(BaseModel):
    """Represents a team member and all their assigned tasks."""
    employee_filename: str
    assignments: List[EmployeeAssignment]

class TeamBuilderOutput(BaseModel):
    """The final output of the team builder node."""
    team: List[TeamMember]
    unassigned_tasks: List[str]
    rationale: str

class TeamBuilderState(TypedDict):
    """The state for the team builder graph."""
    num_employees: int
    project_analysis: Dict[str, Any]
    employee_skills: List[Dict[str, Any]]
    team: Dict[str, Any]


# -----------------------------
# 2. HELPER FUNCTIONS
# -----------------------------
def get_embedding(skill: str) -> np.ndarray:
    """Creates a simple embedding for a skill."""
    # A real implementation would use a pre-trained model like Word2Vec or GloVe,
    # but for this example, we'll use a simple hash-based embedding.
    import hashlib
    return np.frombuffer(hashlib.sha256(skill.encode()).digest(), dtype=np.float32)[:5]

def get_graph_based_score(employee_skills: List[str], task_skills: List[str]) -> float:
    """
    Calculates a score based on the skills graph and embeddings.
    """
    if not task_skills or not employee_skills:
        return 0.0

    score = 0.0
    
    emp_skill_set = set(employee_skills)
    task_skill_set = set(task_skills)

    # 1. Direct matches
    direct_matches = emp_skill_set.intersection(task_skill_set)
    score += len(direct_matches) * 1.0  # Full point for direct match

    remaining_task_skills = task_skill_set - direct_matches
    graph_matched_skills = set()

    # 2. Graph-based matches (parents/children)
    if remaining_task_skills:
        for t_skill in remaining_task_skills:
            parents, children = get_skill_relatives(SKILLS_GRAPH, t_skill)
            related_skills = set()
            if parents:
                related_skills.update(parents)
            if children:
                related_skills.update(children)
            
            if emp_skill_set.intersection(related_skills):
                score += 0.5  # Half point for related skill
                graph_matched_skills.add(t_skill)
                    
    # 3. Embedding-based similarity for remaining skills
    embedding_skills_to_check = remaining_task_skills - graph_matched_skills
    unmatched_emp_skills = emp_skill_set - direct_matches

    if embedding_skills_to_check and unmatched_emp_skills:
        task_embeddings = np.array([get_embedding(s) for s in embedding_skills_to_check])
        emp_embeddings = np.array([get_embedding(s) for s in unmatched_emp_skills])
        
        # Compute cosine similarity
        try:
            # Normalize embeddings to unit vectors for cosine similarity
            task_embeddings_norm = task_embeddings / np.linalg.norm(task_embeddings, axis=1, keepdims=True)
            emp_embeddings_norm = emp_embeddings / np.linalg.norm(emp_embeddings, axis=1, keepdims=True)
            
            cosine_sim = np.dot(task_embeddings_norm, emp_embeddings_norm.T)

            # Add the max similarity for each task skill to the score
            if cosine_sim.size > 0:
                max_sim_scores = np.max(cosine_sim, axis=1)
                score += np.sum(max_sim_scores)
        except (ValueError, ZeroDivisionError):
            # Handle cases with zero-length vectors if they occur
            pass
            
    # Normalize score
    return score / len(task_skill_set) if task_skill_set else 0.0

def is_employee_available(schedule: List[Tuple[int, int]], task_start: int, task_duration: int) -> bool:
    """Checks if an employee is available for a task."""
    task_end = task_start + task_duration
    for busy_start, busy_end in schedule:
        if task_start < busy_end and task_end > busy_start:
            return False  # Overlap
    return True

# -----------------------------
# 3. TEAM BUILDER NODE
# -----------------------------

def team_builder_node(state: TeamBuilderState) -> Dict[str, Any]:
    """
    Analyzes project requirements and available employee skills to form the best possible team,
    allowing one employee to handle multiple tasks sequentially.
    """
    print("--- NODE: TEAM BUILDER ---")
    project_analysis_file = "project_analysis.json"
    skills_summary_file = "skill_results/all_skills_summary.json"
    output_filename = "team_composition.json"

    # --- Load and Validate Inputs ---
    try:
        with open(project_analysis_file, "r", encoding="utf-8") as f:
            project_data = json.load(f)
        project_analysis = ProjectAnalysisOutput(**project_data)
        project_tasks = project_analysis.tasks
        print(f"✅ Loaded and validated {project_analysis_file}")
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"❌ Error loading or validating {project_analysis_file}: {e}")
        return {"team": {"team": [], "unassigned_tasks": [], "rationale": f"Failed to load project analysis: {e}"}}

    try:
        with open(skills_summary_file, "r", encoding="utf-8") as f:
            skills_data = json.load(f)
        employee_skills_data = AllEmployeeSkills.model_validate(skills_data).root
        print(f"✅ Loaded and validated {skills_summary_file}")
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"❌ Error loading or validating {skills_summary_file}: {e}")
        return {"team": {"team": [], "unassigned_tasks": [], "rationale": f"Failed to load employee skills: {e}"}}

    # --- 1. Prepare Employee Data ---
    employees = []
    for emp in employee_skills_data:
        emp_skill_list = []
        for category in emp.summary.values():
            emp_skill_list.extend([skill.lower() for skill in category])
        employees.append({
            "filename": emp.filename,
            "skills": list(set(emp_skill_list)),
            "schedule": []  # List of (start_day, end_day) tuples
        })

    # --- 2. Sort tasks by start day to handle dependencies ---
    sorted_tasks = sorted(project_tasks, key=lambda t: t.start_days_from_kickoff)
    
    # --- 3. Assign tasks using a greedy approach ---
    assignments: Dict[str, List[EmployeeAssignment]] = {emp["filename"]: [] for emp in employees}
    unassigned_tasks = []

    for task in sorted_tasks:
        best_employee = None
        max_score = -1

        task_skills = [skill.lower() for skill in task.skills]
        task_start = task.start_days_from_kickoff
        task_duration = task.duration_days

        for emp in employees:
            if is_employee_available(emp["schedule"], task_start, task_duration):
                score = get_graph_based_score(emp["skills"], task_skills)
                if score > max_score:
                    max_score = score
                    best_employee = emp
        
        if best_employee and max_score >= 0.5:  # Set a minimum score threshold
            # Assign task
            task_end = task_start + task_duration
            best_employee["schedule"].append((task_start, task_end))
            
            emp_skills_set = set(best_employee["skills"])
            task_skills_set = set(task_skills)
            
            assignment = EmployeeAssignment(
                task_name=task.name,
                start_day=task_start,
                end_day=task_end,
                skills_match=list(emp_skills_set.intersection(task_skills_set)),
                missing_skills=list(task_skills_set - emp_skills_set)
            )
            assignments[best_employee["filename"]].append(assignment)
        else:
            unassigned_tasks.append(task.name)

    # --- 4. Format the output ---
    team_members = [
        TeamMember(employee_filename=emp_filename, assignments=emp_assignments)
        for emp_filename, emp_assignments in assignments.items() if emp_assignments
    ]
    
    # --- 5. Validate and Save Output ---
    try:
        rationale = (
            f"Formed a team of {len(team_members)} based on semantic skill matching and availability. "
            f"{len(unassigned_tasks)} tasks could not be assigned."
        )
        team_composition = TeamBuilderOutput(
            team=team_members,
            unassigned_tasks=unassigned_tasks,
            rationale=rationale,
        )
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(team_composition.model_dump(), f, indent=2)
        print(f"✅ Team composition saved to {output_filename}")
        
    except Exception as e:
        error_payload = {"team": [], "unassigned_tasks": [t.name for t in project_tasks], "rationale": f"Validation failed for team output: {e}"}
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(error_payload, f, indent=2)
        print(f"❌ Error saving team composition. Details saved to {output_filename}")
        return {"team": error_payload}

    return {"team": team_composition.model_dump()}

