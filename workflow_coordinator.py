import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

class InterviewWorkflow:
    def __init__(self, jd_file_path: str):
        self.jd_file_path = jd_file_path
        self.output_dir = Path("interview_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    async def extract_job_description(self) -> Dict:
        """Extract job description details using jd_with_json.py"""
        from fastapi.testclient import TestClient
        from jd_with_json import app
        
        client = TestClient(app)
        with open(self.jd_file_path, 'rb') as f:
            response = client.post("/extract/upload-sync", files={"file": f})
            
        if response.status_code != 200:
            raise Exception(f"Failed to extract job description: {response.text}")
            
        return response.json()["job_details"]
    
    def generate_questions(self, job_details: Dict) -> List[str]:
        """Generate interview questions based on job details"""
        # Save job details for questions.py to use
        with open("job_details.json", "w") as f:
            json.dump(job_details, f, indent=2)
            
        # Import and run questions generator
        from questions import HRQuestioningSession
        
        session = HRQuestioningSession(
            job_details=job_details,
            total_questions=5  # Customize number of questions
        )
        
        return [q["question"] for q in session.get_session_summary()["questions_and_answers"]]
    
    async def conduct_interview(self, questions: List[str]) -> List[Dict]:
        """Conduct the interview using voice_pipeline.py"""
        from voice_pipeline import ask_and_listen
        from db import save_transcript
        
        results = []
        for question in questions:
            print(f"\nAsking question: {question}")
            answer = await ask_and_listen(question)
            save_transcript(question, answer)
            results.append({"question": question, "answer": answer})
            
        return results
    
    def save_interview_results(self, job_details: Dict, results: List[Dict]):
        """Save complete interview results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"interview_results_{timestamp}.json"
        
        complete_results = {
            "job_details": job_details,
            "interview_results": results,
            "timestamp": timestamp
        }
        
        with open(output_file, "w") as f:
            json.dump(complete_results, f, indent=2)
            
        return output_file
    
    async def run(self):
        """Execute complete interview workflow"""
        try:
            print("1. Extracting job description...")
            job_details = await self.extract_job_description()
            
            print("\n2. Generating interview questions...")
            questions = self.generate_questions(job_details)
            
            print("\n3. Starting interview...")
            results = await self.conduct_interview(questions)
            
            print("\n4. Saving results...")
            output_file = self.save_interview_results(job_details, results)
            
            print(f"\n✅ Interview completed! Results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error during workflow: {e}")
            raise