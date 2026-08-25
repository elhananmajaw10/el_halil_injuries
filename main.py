from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import json


# -----------------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------------

load_dotenv()


# -----------------------------------
# FASTAPI APP
# -----------------------------------

app = FastAPI(
    title="Football Injury AI API",
    description="Backend connecting ExerciseDB and Ollama",
    version="1.0.0"
)


# -----------------------------------
# CORS CONFIGURATION
# -----------------------------------

# Allows your local frontend (Vite) to communicate
# with this FastAPI backend.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

EXERCISE_API_URL = "https://oss.exercisedb.dev/api/v1/exercises"


# -----------------------------------
# REQUEST MODEL
# -----------------------------------

class InjuryRequest(BaseModel):
    message: str


# -----------------------------------
# BASIC ENDPOINT
# -----------------------------------

@app.get("/")
def home():
    return {
        "message": "Football Injury AI Backend is running"
    }


# -----------------------------------
# HEALTH CHECK
# -----------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ollama_url": OLLAMA_URL
    }


# -----------------------------------
# SEARCH EXERCISES
# -----------------------------------

@app.get("/exercises/search")
def search_exercises(
    query: str,
    threshold: float = 0.5
):
    if not RAPIDAPI_KEY:
        raise HTTPException(
            status_code=500,
            detail="RAPIDAPI_KEY is not configured."
        )

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "Accept": "application/json"
    }

    params = {
        "search": query,
        "threshold": threshold
    }

    try:
        response = requests.get(
            f"{EXERCISE_API_URL}/search",
            headers=headers,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"ExerciseDB API error: {str(e)}"
        )


# -----------------------------------
# GET EXERCISE DETAILS
# -----------------------------------

@app.get("/exercises/{exercise_id}")
def get_exercise_details(exercise_id: str):

    if not RAPIDAPI_KEY:
        raise HTTPException(
            status_code=500,
            detail="RAPIDAPI_KEY is not configured."
        )

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "Accept": "application/json"
    }

    try:
        response = requests.get(
            f"{EXERCISE_API_URL}/{exercise_id}",
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"ExerciseDB API error: {str(e)}"
        )


# -----------------------------------
# AI ROUTER + EXERCISE DETAILS
# -----------------------------------

@app.post("/injury/analyze")
def analyze_injury(request: InjuryRequest):

    # -----------------------------------
    # VALIDATE OLLAMA URL
    # -----------------------------------

    if not OLLAMA_URL:
        raise HTTPException(
            status_code=500,
            detail="OLLAMA_URL is not configured."
        )

    # -----------------------------------
    # STEP 1: CREATE ROUTER PROMPT
    # -----------------------------------

    router_prompt = f"""
You are an API routing assistant for a football exercise and recovery application.

Analyze the user's message and determine the most relevant exercise
search keyword.

Your job is ONLY to identify the relevant exercise/body-area keyword
for searching the ExerciseDB database.

Examples:

User: "I have pain in my hamstring after football"

Response:
{{
    "action": "search_exercises",
    "query": "hamstring"
}}

User: "I want exercises for my calf"

Response:
{{
    "action": "search_exercises",
    "query": "calf"
}}

User: "My quadriceps feels tight"

Response:
{{
    "action": "search_exercises",
    "query": "quad"
}}

User message:
"{request.message}"

Return ONLY valid JSON.

Rules:
- Do not include explanations.
- Do not use markdown.
- Do not include ```json.
- The action must be "search_exercises".
- The query must be a short ExerciseDB-friendly search term.
"""


    try:

        # -----------------------------------
        # STEP 2: ASK OLLAMA
        # -----------------------------------

        ollama_response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "llama3.2",
                "prompt": router_prompt,
                "stream": False,
                "format": "json"
            },
            timeout=120
        )

        ollama_response.raise_for_status()

        ollama_data = ollama_response.json()

        router_result = json.loads(
            ollama_data["response"]
        )

        action = router_result.get("action")
        query = router_result.get("query")

        # -----------------------------------
        # VALIDATE OLLAMA RESPONSE
        # -----------------------------------

        if action != "search_exercises" or not query:
            raise HTTPException(
                status_code=400,
                detail="Could not determine a valid exercise search query."
            )

        # -----------------------------------
        # STEP 3: SEARCH EXERCISEDB
        # -----------------------------------

        if not RAPIDAPI_KEY:
            raise HTTPException(
                status_code=500,
                detail="RAPIDAPI_KEY is not configured."
            )

        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "Accept": "application/json"
        }

        params = {
            "search": query,
            "threshold": 0.5
        }

        search_response = requests.get(
            f"{EXERCISE_API_URL}/search",
            headers=headers,
            params=params,
            timeout=15
        )

        search_response.raise_for_status()

        search_data = search_response.json()

        exercises = search_data.get("data", [])


        # -----------------------------------
        # NO EXERCISES FOUND
        # -----------------------------------

        if not exercises:
            return {
                "user_message": request.message,
                "router_decision": router_result,
                "message": "No matching exercises found.",
                "exercises": []
            }


        # -----------------------------------
        # STEP 4: SELECT FIRST EXERCISE
        # -----------------------------------

        first_exercise = exercises[0]

        exercise_id = first_exercise.get("exerciseId")

        if not exercise_id:
            raise HTTPException(
                status_code=500,
                detail="Exercise ID was not found in the search response."
            )


        # -----------------------------------
        # STEP 5: GET FULL EXERCISE DETAILS
        # -----------------------------------

        details_response = requests.get(
            f"{EXERCISE_API_URL}/{exercise_id}",
            headers=headers,
            timeout=15
        )

        details_response.raise_for_status()

        exercise_details = details_response.json()


        # -----------------------------------
        # STEP 6: RETURN RESULT
        # -----------------------------------

        return {
            "user_message": request.message,
            "router_decision": router_result,
            "exercise": exercise_details
        }


    # -----------------------------------
    # ERROR HANDLING
    # -----------------------------------

    except HTTPException:
        raise

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="A request to Ollama or ExerciseDB timed out."
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"External API error: {str(e)}"
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Ollama returned invalid JSON."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )