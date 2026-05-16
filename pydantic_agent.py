from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field

load_dotenv()

model= OpenAIChatModel(
    model_name= "openai/gpt-oss-20b",
    provider= OpenAIProvider(
        api_key= os.getenv("GROQ_API_KEY"),
        base_url= "https://api.groq.com/openai/v1"
    ),
)

# response = client.responses.create(
#     input="Explain the importance of fast language models",
#     model="openai/gpt-oss-20b",
# )
# print(response.output_text)

agent= Agent(
    model, 
    instructions="You are a concise assistant. Answer in one or two sentences.",
)

# result= agent.run_sync("what is a large language model?")
# print(result.output)

### getting structured, validated outputs
class JobPosting(BaseModel):
    job_title: str
    company_name: str
    required_skills: list[str] =Field(description ="Technical skills explicitly stated")
    seniority_level: str =Field(description= "e.g. Junior, Mid-level, Senior, Lead")
    is_remote: bool

agent=Agent(
    model,
    output_type=JobPosting,
    instructions="Extract structured job posting information. Only include what is explicitly stated."
)

result= agent.run_sync("""
    We are hiring a Senior Python Engineer at CoolData Inc. The role is fully remote.
    Required: FastAPI, PostgreSQL, Docker. Kubernetes is a plus.
    """)

posting=result.output
print(posting.job_title, posting.seniority_level, posting.is_remote)
print()
print(posting.model_dump())
