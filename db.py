from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["voice_transcripts"]
collection = db["answers"]

def save_transcript(question: str, answer: str):
    collection.insert_one({"question": question, "answer": answer})
