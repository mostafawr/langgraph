#!/usr/bin/env python3
"""
Runs the CV skill extractor pipeline.
"""
import os
import json
from chains.extractor_with_langgraph import app, load_cv_text, CV_FOLDER, GraphState

def main():
    """
    Main function to run the skill extraction workflow.
    """
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
