import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner
 
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
# Static HR Questions - These remain the same for all candidates
STATIC_HR_QUESTIONS = [
    "Tell me about yourself and walk me through your professional background.",
   
    "What interests you most about this position and why do you want to work for our company?",
   
    "Describe a challenging project you worked on recently. What was your role and how did you overcome the obstacles?",
   
    "How do you handle working under pressure and tight deadlines? Can you give me a specific example?",
   
    "What are your greatest strengths and how do they relate to this role?",
   
    "Tell me about a time when you had to work with a difficult team member. How did you handle the situation?",
   
    "Where do you see yourself in 3-5 years, and how does this position align with your career goals?",
   
    "What do you consider to be your areas for improvement, and what steps are you taking to address them?",
   
    "Describe a situation where you had to learn a new skill or technology quickly. How did you approach it?",
   
    "Do you have any questions about the role, our company culture, or what success looks like in this position?"
]
 
class HRQuestioningSession:
    def __init__(self, candidate_name=None, job_details=None, total_questions=5):
        self.candidate_name = candidate_name
        self.job_details = job_details
        self.total_questions = min(total_questions, len(STATIC_HR_QUESTIONS))  # Ensure we don't exceed available questions
        self.current_question_number = 0
        self.questions_asked = []
        self.candidate_answers = []
        self.session_start_time = datetime.now()
        self.selected_questions = STATIC_HR_QUESTIONS[:self.total_questions]  # Use first N questions
   
    async def ask_next_question(self):
        """Ask the next static question"""
       
        if self.current_question_number >= self.total_questions:
            print("\n✅ All questions completed!")
            return None
       
        # Get the static question
        question = self.selected_questions[self.current_question_number]
       
        print(f"\n🎯 Question {self.current_question_number + 1}/{self.total_questions}")
       
        # Store the question
        self.questions_asked.append({
            "question_number": self.current_question_number + 1,
            "question": question,
            "asked_at": datetime.now().isoformat()
        })
       
        print(f"\n👩‍💼 HR QUESTION {self.current_question_number + 1}:")
        print("-" * 40)
        print(question)
        print("\n⏳ Waiting for candidate response...")
       
        self.current_question_number += 1
        return question
   
    def store_candidate_answer(self, answer):
        """Store candidate's answer without processing it"""
       
        if not answer.strip():
            print("⚠️  Empty answer received. Please provide a response.")
            return False
       
        # Simply store the answer
        self.candidate_answers.append({
            "question_number": self.current_question_number,
            "answer": answer,
            "answered_at": datetime.now().isoformat()
        })
       
        print("✅ Answer recorded. Moving to next question...\n")
        return True
   
    def get_session_summary(self):
        """Get summary of the questioning session"""
       
        return {
            "session_info": {
                "candidate_name": self.candidate_name,
                "job_details": self.job_details,
                "total_questions": self.total_questions,
                "questions_completed": len(self.questions_asked),
                "answers_received": len(self.candidate_answers),
                "session_duration": str(datetime.now() - self.session_start_time),
                "session_date": self.session_start_time.isoformat()
            },
            "questions_and_answers": [
                {
                    "question_number": i + 1,
                    "question": self.questions_asked[i]["question"] if i < len(self.questions_asked) else "Not asked",
                    "answer": self.candidate_answers[i]["answer"] if i < len(self.candidate_answers) else "Not answered"
                }
                for i in range(self.total_questions)
            ]
        }
   
    def save_session_data(self, filename=None):
        """Save complete session data"""
       
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            candidate_suffix = f"_{self.candidate_name.replace(' ', '_')}" if self.candidate_name else ""
            filename = f"hr_session{candidate_suffix}_{timestamp}.json"
       
        try:
            import json
            session_data = self.get_session_summary()
           
            with open(filename, "w") as f:
                json.dump(session_data, f, indent=2)
           
            print(f"💾 Session data saved to {filename}")
            return filename
           
        except Exception as e:
            print(f"❌ Error saving session: {e}")
            return None
   
    def display_all_questions(self):
        """Display all questions that will be asked in this session"""
        print("\n📋 QUESTIONS FOR THIS SESSION:")
        print("-" * 50)
        for i, question in enumerate(self.selected_questions, 1):
            print(f"{i}. {question}")
        print("-" * 50)
 
# Job details (can be customized for different positions)
job_details = {
    "position": "Senior DevOps Engineer",
    "company": "Tech Solutions Inc",
    "requirements": ["AWS", "Kubernetes", "CI/CD", "Infrastructure Automation"],
    "experience_level": "3-5 years",
    "location": "Pune"
}
 
async def run_hr_session(candidate_name="Candidate", num_questions=5):
    """Run a complete HR questioning session with static questions"""
   
    print("🚀 HR QUESTIONING SESSION")
    print("=" * 50)
    print(f"Candidate: {candidate_name}")
    print(f"Position: {job_details['position']}")
    print(f"Company: {job_details['company']}")
    print(f"Number of Questions: {num_questions}")
    print("=" * 50)
   
    # Initialize session
    session = HRQuestioningSession(candidate_name, job_details, total_questions=num_questions)
   
    # Optionally display all questions
    session.display_all_questions()
   
    # Sample candidate responses (in real scenario, these would be user inputs)
    sample_answers = [
        "I'm a DevOps engineer with over 3 years of experience in cloud infrastructure and automation. I started my career as a System Administrator at TCS, where I learned Linux systems and automation. Then I moved to Infosys as a DevOps Engineer, where I've been working with AWS, Docker, and Kubernetes to build scalable deployment pipelines.",
       
        "I'm particularly excited about this Senior DevOps Engineer role because it combines my technical skills with the opportunity to architect scalable solutions. Your company's focus on cloud-native technologies and DevOps best practices aligns perfectly with my experience and interests. I'm also drawn to the collaborative culture and the chance to mentor junior team members.",
       
        "Recently, I led a cloud migration project where we had to move 15 legacy applications from on-premise to AWS within a tight 3-month deadline. The main challenge was maintaining zero downtime during migration. I created a phased migration strategy using blue-green deployment techniques and extensive testing environments. We successfully completed the migration on time and achieved 40% better performance.",
       
        "I actually thrive under pressure because it brings out my problem-solving skills. During a critical production outage last year, our entire CI/CD pipeline was down during a major release week. I quickly assembled a war room, identified the root cause in our Jenkins configuration, and implemented a temporary workaround within 2 hours. I then worked with the team to implement a permanent fix and better monitoring to prevent similar issues.",
       
        "My greatest strength is my ability to bridge the gap between development and operations teams. I have strong technical skills in automation and cloud technologies, but I also excel at communication and collaboration. For example, I created documentation and training sessions that helped our development team understand our deployment processes, which reduced deployment-related issues by 60%.",
       
        "I had a situation where a team member was consistently resistant to adopting our new containerization standards. Instead of escalating immediately, I took time to understand their concerns - they were worried about the learning curve and job security. I offered to pair-program with them and provided additional training resources. Eventually, they became one of our strongest advocates for the new approach.",
       
        "In 3-5 years, I see myself in a DevOps Architect role where I can design and implement enterprise-scale infrastructure solutions. I want to be involved in strategic technology decisions and help organizations adopt cloud-native practices. This position would give me the senior-level experience and exposure to complex systems that I need to achieve those goals.",
       
        "I sometimes focus too much on technical perfection and can spend more time than necessary on optimization. I'm working on this by setting clearer time boundaries for tasks and learning to balance 'good enough' with 'perfect.' I've started using time-boxing techniques and regularly check in with stakeholders to ensure I'm delivering value efficiently.",
       
        "When Kubernetes was becoming essential for our infrastructure, I had to learn it quickly for a project with a tight deadline. I created a structured learning plan: started with official documentation, built a lab environment, took online courses, and joined the Kubernetes community forums. I also found a mentor who was already experienced with K8s. Within 3 weeks, I was confident enough to lead the Kubernetes implementation for our project.",
       
        "I'd like to know more about the team structure and how the DevOps team collaborates with other departments. Also, what are the biggest infrastructure challenges the company is currently facing? And what opportunities are there for professional development and staying current with emerging technologies?"
    ]
   
    # Conduct the questioning session
    for i in range(session.total_questions):
        # Ask question
        question = await session.ask_next_question()
        if not question:
            break
       
        # Wait a moment (simulating thinking time)
        await asyncio.sleep(1)
       
        # Get candidate answer (in real app, this would be user input)
        if i < len(sample_answers):
            candidate_answer = sample_answers[i]
            print(f"👤 CANDIDATE RESPONSE:")
            print(f"{candidate_answer}")
           
            # Store answer
            session.store_candidate_answer(candidate_answer)
        else:
            print("⏳ Waiting for candidate response...")
            break
   
    # Display session summary
    print("\n📊 SESSION SUMMARY")
    print("=" * 50)
    summary = session.get_session_summary()
    print(f"Candidate: {summary['session_info']['candidate_name']}")
    print(f"Questions Asked: {summary['session_info']['questions_completed']}")
    print(f"Answers Received: {summary['session_info']['answers_received']}")
    print(f"Session Duration: {summary['session_info']['session_duration']}")
   
    # Save session data
    session.save_session_data()
   
    print("\n✅ HR Questioning Session Completed!")
    return session
 
# Interactive function to run session with custom parameters
async def run_custom_session():
    """Run a session with custom candidate name and question count"""
   
    print("🎯 CUSTOM HR SESSION SETUP")
    print("-" * 30)
   
    # In a real application, you would get these inputs from user
    candidate_name = "Rajesh Patel"  # input("Enter candidate name: ")
    num_questions = 5  # int(input(f"Enter number of questions (1-{len(STATIC_HR_QUESTIONS)}): "))
   
    if num_questions > len(STATIC_HR_QUESTIONS):
        num_questions = len(STATIC_HR_QUESTIONS)
        print(f"⚠️  Maximum {len(STATIC_HR_QUESTIONS)} questions available. Using {num_questions} questions.")
   
    session = await run_hr_session(candidate_name, num_questions)
    return session
 
async def main():
    """Main execution function"""
    print("🏢 HR INTERVIEW SYSTEM - STATIC QUESTIONS")
    print("=" * 60)
   
    # Run the session
    await run_custom_session()
   
    print("\n📚 AVAILABLE STATIC QUESTIONS:")
    print("-" * 40)
    for i, question in enumerate(STATIC_HR_QUESTIONS, 1):
        print(f"{i:2d}. {question}")
 
if __name__ == "__main__":
    asyncio.run(main())
 