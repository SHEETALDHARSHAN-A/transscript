import asyncio
import argparse
from workflow_coordinator import InterviewWorkflow
from pathlib import Path

async def main():
    parser = argparse.ArgumentParser(description="Run automated interview process")
    parser.add_argument(
        "jd_file",
        type=str,
        help="Path to job description file (PDF/DOCX)"
    )
    args = parser.parse_args()
    
    jd_path = Path(args.jd_file)
    if not jd_path.exists():
        print(f"Error: File not found: {jd_path}")
        return
        
    print(f"Starting interview workflow with JD: {jd_path}")
    workflow = InterviewWorkflow(str(jd_path))
    await workflow.run()

if __name__ == "__main__":
    asyncio.run(main())