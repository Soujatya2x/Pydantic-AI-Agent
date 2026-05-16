import json
import os
from pprint import pprint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider 

load_dotenv()
model=OpenAIChatModel(
    model_name="openai/gpt-oss-20b",
    provider=OpenAIProvider(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
)

NUTRITION_DB={
    "chicken breast": {"calories": 165, "protein_g": 31, "carbs_g": 0,  "fat_g": 3.6},
    "brown rice":     {"calories": 216, "protein_g": 5,  "carbs_g": 45, "fat_g": 1.8},
    "broccoli":       {"calories": 55,  "protein_g": 3.7,"carbs_g": 11, "fat_g": 0.6},
    "olive oil":      {"calories": 119, "protein_g": 0,  "carbs_g": 0,  "fat_g": 13.5},
}

class MealSummary(BaseModel):
    total_calories: float
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    health_verdict: str
    recommendation: str

agent= Agent(
    model,
    output_type= MealSummary,
    instructions= "Use tools to look up ingredient data, compute totals, and give a verdict."
)

@agent.tool_plain
def get_ingredient_nutrition(ingredient: str) ->str:
    """
    Look up calories, protein, carbs, and fat per 100g for a single ingredient.
    Returns an error message if the ingredient is not found in the database.
    """
    data= NUTRITION_DB.get(ingredient.lower().strip())
    if data: 
        return json.dumps({"ingredient": ingredient, **data})
    return f"Not found. Available: {', '.join(NUTRITION_DB)}" 

result= agent.run_sync(
    "Analyse: 200g chicken breast, 150g brown rice, 100g broccoli, 10g olive oil."
)
pprint(result.output.model_dump())