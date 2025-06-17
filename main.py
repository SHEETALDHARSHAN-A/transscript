from fastapi import FastAPI
import asyncio
from voice_pipeline import ask_and_listen
from db import save_transcript
import json

app = FastAPI()

def load_questions():
    with open("questions.json") as f:
        return json.load(f)

@app.get("/interview")
async def interview():
    questions = load_questions()
    results = []

    for question in questions:
        answer = await ask_and_listen(question)
        save_transcript(question, answer)
        results.append({"question": question, "answer": answer})

    return {"status": "complete", "data": results}
