import os
from fastapi import FastAPI
from pydantic import BaseModel
from crew import run_crew
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="学生规划智能体API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)

class RequestBody(BaseModel):
    major: str
    grade: str
    skills: str
    goal: str
    city: str = "未知"

@app.post("/api/generate_plan")
async def generate_plan(req: RequestBody):
    input_data = req.model_dump()
    try:
        loop = asyncio.get_event_loop()
        crew_result = await loop.run_in_executor(executor, run_crew, input_data)
        result_text = crew_result
        return {"result": result_text}
    except Exception as e:
        print("====接口内部异常====")
        print(repr(e))
        raise e

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
