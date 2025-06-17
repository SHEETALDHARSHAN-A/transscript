import sounddevice as sd
import numpy as np
import asyncio
from agents import AssistantAgent
from agents.voice import VoicePipeline, SingleAgentVoiceWorkflow
import tempfile

agent = AssistantAgent(name="Interviewer", instructions="Listen to the user and transcribe their answers.")
pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))

async def ask_and_listen(question: str, max_duration: int = 120) -> str:
    print(f"Asking: {question}")

    # Speak the question using TTS
    async for event in (await pipeline.run(question)).stream():
        if event.type == "audio":
            sd.play(event.data, samplerate=24000)
            sd.wait()

    # Record user response
    print("Listening for answer (up to 2 minutes)...")
    audio = sd.rec(int(max_duration * 24000), samplerate=24000, channels=1, dtype=np.float32)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        np.save(tmpfile, audio)
        tmpfile.flush()
        result = await pipeline.run(audio_input=tmpfile.name)

    full_text = ""
    async for event in result.stream():
        if event.type == "text":
            print("Transcript:", event.data.strip())
            full_text += event.data.strip() + " "

    return full_text.strip()
