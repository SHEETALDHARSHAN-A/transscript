# Automated Interview System

This system automates the interview process by:
1. Extracting job requirements from a job description
2. Generating relevant interview questions
3. Conducting a voice interview
4. Saving transcripts and results

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure MongoDB is running for transcript storage:
```bash
mongod
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Run the complete interview workflow:
```bash
python run_interview.py path/to/job_description.pdf
```

2. Or run individual components:

- Extract job description:
```bash
python jd_with_json.py
```

- Generate questions:
```bash
python questions.py
```

- Run interview:
```bash
python main.py
```

## Output

Results are saved in the `interview_outputs` directory containing:
- Job description details
- Generated questions
- Interview transcripts
- Complete interview results

## Directory Structure

```
transcription_making/
├── workflow_coordinator.py   # Main workflow coordinator
├── run_interview.py         # CLI interface
├── jd_with_json.py         # Job description extractor
├── questions.py            # Question generator
├── voice_pipeline.py       # Voice interaction
├── db.py                   # MongoDB interface
├── main.py                 # FastAPI server
└── interview_outputs/      # Results directory
```