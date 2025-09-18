import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

load_dotenv()
YOUR_OPENAI_API_KEY = os.getenv("YOUR_OPENAI_API_KEY")
CONFIG_FILE = "S/ocean.json"

# Global MCP client + agent
client = MCPClient.from_config_file(CONFIG_FILE)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=YOUR_OPENAI_API_KEY)
agent = MCPAgent(llm=llm, client=client, max_steps=15, memory_enabled=True,verbose=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if client and client.sessions:
        await client.close_all_sessions()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    user_input: str

@app.post("/chat")
async def chat(query: Query):
    response = await agent.run((query.user_input).split("Final Answer:")[-1].strip())
    return {"response": response}