import os
import sys
import json
import random
import uuid
from datetime import datetime
from google import genai
from google.genai import types

# ==========================================
# CONFIG
# ==========================================
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("Missing GEMINI_API_KEY")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

POPULATION_SIZE = 5
GENERATIONS = 4

BOARDROOM_DIR = "boardrooms"
os.makedirs(BOARDROOM_DIR, exist_ok=True)
LOG_FILE = os.path.join(BOARDROOM_DIR, "evolution_lowquota.jsonl")

# ==========================================
# LOGGING
# ==========================================
def log(event, data):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.now().isoformat(),
            "event": event,
            "data": data
        }, ensure_ascii=False) + "\n")

# ==========================================
# LLM CALL
# ==========================================
def call_llm(prompt, system="", temp=0.3):
    try:
        res = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temp,
            ),
        )
        return res.text
    except Exception as e:
        print(f"API Error: {e}")
        return ""

# ==========================================
# INITIAL POPULATION
# ==========================================
def seed_population():
    prompt = """
Generate 5 diverse deep-tech startup ideas for a solo AI researcher.
Each idea should be different and monetizable.
Return as JSON list of strings.
"""

    raw = call_llm(prompt, temp=0.9)

    try:
        ideas = json.loads(raw)
    except:
        ideas = [
            f"Idea fallback {i}: {raw[:50]}"
            for i in range(5)
        ]

    return [
        {
            "id": str(uuid.uuid4())[:8],
            "text": i,
            "score": 0
        }
        for i in ideas[:POPULATION_SIZE]
    ]

# ==========================================
# EVOLUTION STEP (CORE MAGIC)
# ==========================================
def evolve(population, generation):
    print(f"\n=== GENERATION {generation} ===")

    prompt = f"""
You are an evolutionary startup engine.

Given these ideas:

{json.dumps([p['text'] for p in population], indent=2)}

DO THE FOLLOWING:

1. Score each idea (1–10)
2. Keep best ideas
3. Mutate weak ideas into better ones
4. Create 1–2 crossover ideas
5. Return EXACT JSON:

{{
  "evaluated": [
    {{"text": "...", "score": 8}},
  ],
  "next_generation": [
    "idea1",
    "idea2",
    "idea3",
    "..."
  ]
}}

Rules:
- Keep exactly {POPULATION_SIZE} ideas
- Make ideas more technical and monetizable
- Prefer deep-tech AI startups
"""

    response = call_llm(prompt, temp=0.4)

    log("raw_generation", response)

    try:
        data = json.loads(response)
        next_gen = data["next_generation"]
        evaluated = data["evaluated"]
    except:
        print("Parse error, fallback triggered")
        next_gen = [p["text"] for p in population]
        evaluated = [{"text": p["text"], "score": 5} for p in population]

    new_population = []
    for idea in next_gen[:POPULATION_SIZE]:
        new_population.append({
            "id": str(uuid.uuid4())[:8],
            "text": idea,
            "score": 0
        })

    print("\nTop evaluated ideas:")
    for e in evaluated:
        print(f"- {e.get('score', 0)} | {e.get('text', '')[:80]}")

    log("evaluated", evaluated)

    return new_population

# ==========================================
# RUN
# ==========================================
def run():
    print("=== LOW QUOTA EVOLUTION ENGINE START ===")

    population = seed_population()
    log("seed", population)

    for gen in range(GENERATIONS):
        population = evolve(population, gen)

    print("\n=== FINAL IDEAS ===\n")

    for i, idea in enumerate(population, 1):
        print(f"\n#{i}")
        print(idea["text"])
        print("-" * 60)

    log("final", population)

# ==========================================
if __name__ == "__main__":
    run()