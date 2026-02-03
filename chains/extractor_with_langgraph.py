"CV skill extractor pipeline built with LangGraph."

# -----------------------------
# 1. CONFIG & SETUP
# -----------------------------
from .common_imports import (
    Any,
    BaseModel,
    ChatPromptTemplate,
    Document,
    END,
    Field,
    List,
    PdfReader,
    StateGraph,
    TypedDict,
    cast,
    json,
    load_env,
    make_llm,
    os,
)

# Load environment variables (API keys, etc.)
load_env()

# Initialize ChatGroq (shared instance, zero temperature for determinism)
llm = make_llm()

CV_FOLDER = "./cvs"


# -----------------------------
# 2. DATA SCHEMAS (Pydantic)
# -----------------------------

class SkillsExtraction(BaseModel):
    """Schema for detailed skills extraction."""
    hard_skills: List[str] = Field(description="Functional capabilities (e.g., Supply Chain Management, Data Analysis, System Design).")
    soft_skills: List[str] = Field(description="Interpersonal traits (e.g., Leadership, Collaboration). Ignore generic fluff.")
    tools_and_tech: List[str] = Field(description="Specific software, hardware, AND programming languages (e.g., Python, Odoo, Excel, AWS).")
    languages: List[str] = Field(description="Spoken/Human languages only (e.g., English, Arabic, German).")

class SkillsSummary(BaseModel):
    """Schema for the final summarized skills."""
    core_hard_skills: List[str] = Field(description="Top 15 most important functional skills (High-level domains).")
    core_soft_skills: List[str] = Field(description="Top 8 distinct, high-impact soft skills.")
    core_tools_and_tech: List[str] = Field(description="Top 15 critical tools, frameworks, and programming languages.")
    core_languages: List[str] = Field(description="All spoken human languages found.")
##list soft - technical - hard skills- tools and tech
# -----------------------------
# 3. GRAPH STATE
# -----------------------------

class GraphState(TypedDict):
    cv_text: str
    initial_skills: dict
    refined_skills: dict
    final_summary: dict

# -----------------------------
# 4. NODES (With Specialized Prompts)
# -----------------------------

def extraction_node(state: GraphState):
    print("--- NODE: EXTRACTING SKILLS ---")
    cv_text = state["cv_text"]
    
    # SYSTEM PROMPT: Enforces the separation between "Capability" (Hard Skill) and "Tool" (Tech)
    system_msg = """You are an expert ATS (Applicant Tracking System). Extract skills from the CV with strict categorization:

1. **Hard Skills (Domains/Capabilities):** Abstract professional abilities. 
   - Example: "Supply Chain Management", "Data Analysis", "Web Development", "Accounting".
   - DO NOT put specific tools here.

2. **Tools & Tech (Software/Languages):** Specific instruments used to perform the work.
   - Include: Programming Languages (Python, Java), Software (Odoo, Excel), Frameworks (React, TensorFlow).
   - Normalize names: "Microsoft Excel" -> "Excel", "React.js" -> "React".

3. **Soft Skills:** High-value interpersonal traits.
   - Example: "Leadership", "Collaboration", "Crisis Management".
   - IGNORE fluff like "Hard worker", "Motivated", "Fast learner".

4. **Languages:** Human spoken languages only (English, Arabic).

Return valid JSON."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{cv_text}")
    ])
    
    chain = prompt | llm.with_structured_output(SkillsExtraction)
    result = chain.invoke({"cv_text": cv_text})
    
    # Convert to dict for JSON serialization
    return {"initial_skills": dict(result) if not isinstance(result, dict) else result}

def reflection_node(state: GraphState):
    print("--- NODE: REFLECTING & REFINING ---")
    cv_text = state["cv_text"]
    initial_skills = state["initial_skills"]
    
    # SYSTEM PROMPT: Focuses on cleaning, merging, and canonicalization
    system_msg = """You are a QA Auditor for CV data. Review the extracted skills:
    
1. **Clean & Standardize:** - Merge "Machine Learning" and "ML" -> "Machine Learning".
   - Ensure "Python" is in Tools, not Hard Skills.
   - Ensure "Data Analysis" is in Hard Skills, not Tools.

2. **Remove Noise:**
   - Delete generic soft skills (e.g., "Punctual").
   - Delete job titles or company names if they were mistakenly extracted.

3. **Check Missing:** - If the CV mentions "Django", ensure "Python" is also added if missing.
   - If the CV mentions "Inventory Control", ensure "Inventory Management" is present.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Original CV Text:\n{cv_text}\n\nInitial JSON:\n{initial_json}")
    ])
    
    chain = prompt | llm.with_structured_output(SkillsExtraction)
    
    result = chain.invoke({
        "cv_text": cv_text, 
        "initial_json": json.dumps(initial_skills, ensure_ascii=False)
    })
    
    # Convert to dict for JSON serialization
    return {"refined_skills": dict(result) if not isinstance(result, dict) else result}

def summary_node(state: GraphState):
    print("--- NODE: SUMMARIZING ---")
    refined_skills = state["refined_skills"]
    
    # SYSTEM PROMPT: Final selection of the "Core" skills
    system_msg = """Final Summarization Task:
1. Select the top **Core** skills that define this candidate's professional identity.
2. Prioritize skills that appear most relevant to their most recent roles.
3. Limit counts:
   - Max 15 Core Hard Skills
   - Max 15 Core Tools/Tech
   - Max 8 Core Soft Skills
4. Ensure strictly clean output (no duplicates). 
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Refined Skills:\n{skills_json}")
    ])
    
    chain = prompt | llm.with_structured_output(SkillsSummary)
    
    result = chain.invoke({
        "skills_json": json.dumps(refined_skills, ensure_ascii=False)
    })
    
    # Convert to dict for JSON serialization
    return {"final_summary": dict(result) if not isinstance(result, dict) else result}

# -----------------------------
# 5. BUILD THE GRAPH
# -----------------------------

workflow = StateGraph(GraphState)

workflow.add_node("extract", extraction_node)
workflow.add_node("reflect", reflection_node)
workflow.add_node("summarize", summary_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "reflect")
workflow.add_edge("reflect", "summarize")
workflow.add_edge("summarize", END)

app = workflow.compile()

# -----------------------------
# 6. FILE HELPERS & MAIN
# -----------------------------

def load_cv_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext in (".docx", ".doc"):
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return ""

def main():
    # Create output folder if it doesn't exist
    output_folder = "skill_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if not os.path.isdir(CV_FOLDER):
        print(f"CV folder '{CV_FOLDER}' not found. Creating it...")
        os.makedirs(CV_FOLDER, exist_ok=True)
        print(f"Please add CV files (.pdf, .docx, .txt) to the '{CV_FOLDER}' folder")
        return
    
    files = [f for f in os.listdir(CV_FOLDER) if f.endswith(('.pdf', '.docx', '.doc', '.txt'))]
    
    if not files:
        print(f"No CV files found in '{CV_FOLDER}' folder")
        return
    
    all_results = []
    
    for filename in files:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        output_file = os.path.join(output_folder, f"{filename.rsplit('.', 1)[0]}_skills.json")

        if os.path.exists(output_file):
            print(f"✅ Skills for {filename} already extracted. Loading from cache.")
            with open(output_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            summary = cached_data.get("summary", {})
            all_results.append({
                "filename": filename,
                "summary": summary
            })
            continue

        path = os.path.join(CV_FOLDER, filename)
        cv_text = load_cv_text(path)
        
        if not cv_text:
            print(f"⚠️  Could not extract text from {filename}")
            continue
        
        try:
            # Run the graph
            initial_state: GraphState = {"cv_text": cv_text}  # type: ignore
            final_state = app.invoke(initial_state)
            
            # Extract final summary
            summary = final_state.get("final_summary", {})
            
            # Display results
            print("\n✅ FINAL SUMMARY:")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            
            # Save individual result
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "filename": filename,
                    "summary": summary,
                    "initial_extraction": final_state.get("initial_skills", {}),
                    "refined_extraction": final_state.get("refined_skills", {})
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Saved to: {output_file}")
            
            # Collect for master results file
            all_results.append({
                "filename": filename,
                "summary": summary
            })
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
    
    # Save master results file
    if all_results:
        master_file = os.path.join(output_folder, "all_skills_summary.json")
        with open(master_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"📊 Master results saved to: {master_file}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
