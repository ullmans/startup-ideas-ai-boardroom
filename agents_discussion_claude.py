"""
╔══════════════════════════════════════════════════════════════╗
║          BOARDROOM EVOLUTION ENGINE  v1.1                    ║
║   Multi-Agent Debate  ×  Evolutionary Selection              ║
║   Rate-limit safe for Gemini Free Tier (10 RPM)              ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    export GEMINI_API_KEY="your_key_here"
    python boardroom_evolution.py

Rate-limit strategy:
    Free tier  (default) : ~6s between calls → stays comfortably under 10 RPM
    Paid tier            : set GEMINI_PAID_TIER=1 → 1.2s delay, ~40 RPM
"""

import os
import sys
import json
import time
from datetime import datetime
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ══════════════════════════════════════════════════════════════
#  CONFIG — tweak these before running
# ══════════════════════════════════════════════════════════════

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("❌  Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

# Set GEMINI_PAID_TIER=1 in your env if you have a paid API key
PAID_TIER = os.environ.get("GEMINI_PAID_TIER", "0") == "1"

client = genai.Client(api_key=API_KEY)
MODEL  = "gemini-2.5-flash"

IDEA_POOL_SIZE  = 4   # Ideas in the gene pool
GENERATIONS     = 3   # Full debate + evolution cycles
TURNS_PER_ROUND = 6   # Agent turns per debate (reduced from 8 → saves 6 calls)
CONTEXT_WINDOW  = 5   # Recent messages visible to each agent

# ── Rate limiting ──────────────────────────────────────────────
# Free tier:  10 RPM  → need ≥6s between calls
# Paid tier:  ≥1000 RPM → 1.2s is fine
MIN_DELAY_FREE = 6.5   # seconds — gives a safety margin over 10 RPM
MIN_DELAY_PAID = 1.2   # seconds

CALL_DELAY = MIN_DELAY_PAID if PAID_TIER else MIN_DELAY_FREE

BOARDROOM_DIR = "boardrooms"
os.makedirs(BOARDROOM_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(BOARDROOM_DIR, f"boardroom_evolution_{TIMESTAMP}.txt")


# ══════════════════════════════════════════════════════════════
#  RATE-LIMIT AWARE API WRAPPER
# ══════════════════════════════════════════════════════════════

_last_call_time = 0.0

def call_api(model, contents, config, max_retries=5):
    """
    Single entry point for ALL Gemini API calls.
    - Enforces minimum delay between calls (token bucket style)
    - Exponential backoff on 429 / 503
    """
    global _last_call_time

    wait_on_error = 10  # first retry wait (seconds)

    for attempt in range(max_retries):
        # ── Enforce minimum inter-call delay ──────────────────
        elapsed = time.monotonic() - _last_call_time
        if elapsed < CALL_DELAY:
            sleep_for = CALL_DELAY - elapsed
            print(f"  {DIM}[throttle: waiting {sleep_for:.1f}s]{RESET}", end="\r")
            time.sleep(sleep_for)

        try:
            _last_call_time = time.monotonic()
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as e:
            err = str(e)
            is_rate_limit = any(
                c in err for c in ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]
            )
            if is_rate_limit and attempt < max_retries - 1:
                print(f"  {DIM}[rate-limited on attempt {attempt+1}, backing off {wait_on_error}s…]{RESET}")
                time.sleep(wait_on_error)
                wait_on_error = min(wait_on_error * 2, 120)  # cap at 2 min
            else:
                raise  # Non-retriable error or max retries hit

    raise RuntimeError("Max retries exceeded")


# ══════════════════════════════════════════════════════════════
#  STRUCTURED OUTPUT SCHEMAS
# ══════════════════════════════════════════════════════════════

class AgentTurn(BaseModel):
    response: str = Field(
        description="Your spoken response in the boardroom. Sharp and concise (3-4 sentences max)."
    )
    next_speaker: str = Field(
        description="Key of the agent who should speak next to keep the debate productive."
    )

class IdeaScore(BaseModel):
    idea_index: int   = Field(description="0-based index of the idea in the pool.")
    score:      int   = Field(description="1–10: technical depth + monetizability + solo-founder feasibility.")
    verdict:    str   = Field(description="One of: KEEP | MUTATE | DROP. Followed by one-sentence reason.")

class EvolutionResult(BaseModel):
    scores:        list[IdeaScore] = Field(description="Score every idea in the pool.")
    evolved_ideas: list[str]       = Field(
        description=f"Exactly {IDEA_POOL_SIZE} ideas for the next generation."
    )


# ══════════════════════════════════════════════════════════════
#  ANSI COLORS & LOGGING HELPERS
# ══════════════════════════════════════════════════════════════

AGENT_COLORS = {
    "visionary":      "\033[95m",
    "critic":         "\033[91m",
    "tech_architect": "\033[94m",
    "mlops":          "\033[96m",
    "domain_expert":  "\033[93m",
    "finance":        "\033[92m",
    "moderator":      "\033[97m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

def clr(key: str, text: str) -> str:
    return f"{AGENT_COLORS.get(key, '')}{text}{RESET}"

def section(title: str):
    bar = "═" * 62
    print(f"\n{BOLD}{bar}\n  {title}\n{bar}{RESET}\n")

def log(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ══════════════════════════════════════════════════════════════
#  AGENT CLASS
# ══════════════════════════════════════════════════════════════

class Agent:
    def __init__(self, key: str, name: str, role: str, system_prompt: str):
        self.key  = key
        self.name = name
        self.system_instruction = (
            f"You are in a live startup boardroom. Your role: {role}.\n"
            f"{system_prompt}\n\n"
            "Rules:\n"
            "- React DIRECTLY to the previous speaker. Quote them if useful.\n"
            "- Be concise and sharp: 3–4 sentences maximum.\n"
            "- Always respond in English.\n"
            "- Pick the next speaker who would most productively challenge YOUR point."
        )

    def speak(self, recent_history: str, valid_keys: list[str]) -> tuple[str, str]:
        prompt = (
            f"Recent conversation:\n{recent_history}\n\n"
            f"Your turn. Respond, then pick the next speaker from: {valid_keys}. "
            f"Do NOT pick yourself ({self.key})."
        )
        try:
            res = call_api(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.8,
                    response_mime_type="application/json",
                    response_schema=AgentTurn,
                ),
            )
            
            raw_text = res.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            data       = json.loads(raw_text.strip())
            reply_text = data.get("response", "(silent)")
            next_key   = data.get("next_speaker", "").lower().strip()

            if next_key not in valid_keys or next_key == self.key:
                idx      = valid_keys.index(self.key)
                next_key = valid_keys[(idx + 1) % len(valid_keys)]

            return reply_text, next_key

        except Exception as e:
            idx = valid_keys.index(self.key)
            next_key = valid_keys[(idx + 1) % len(valid_keys)]
            return f"(skipped due to error: {str(e)[:60]})", next_key


# ══════════════════════════════════════════════════════════════
#  AGENT ROSTER
# ══════════════════════════════════════════════════════════════

agents: dict[str, Agent] = {
    "visionary": Agent(
        key="visionary",
        name="The Visionary 🚀",
        role="Radical innovation advocate",
        system_prompt=(
            "Push for bold, non-obvious ideas. If an idea is attacked, propose a creative pivot "
            "that saves its core insight. You believe most 'practical' objections are failures of imagination."
        ),
    ),
    "critic": Agent(
        key="critic",
        name="The Critic 🔪",
        role="Devil's Advocate",
        system_prompt=(
            "Find the fatal flaw in every idea. Focus on: competition from tech giants, "
            "unrealistic GTM assumptions, regulatory risk. Never fully agree with the previous speaker."
        ),
    ),
    "tech_architect": Agent(
        key="tech_architect",
        name="The Tech Architect 🧠",
        role="Deep-tech feasibility evaluator",
        system_prompt=(
            "Evaluate whether the idea requires genuine AI depth: Transformers, SSMs (Mamba), JEPA, diffusion. "
            "Reject anything solvable by wrapping GPT-4 API. Always suggest one concrete architectural improvement."
        ),
    ),
    "mlops": Agent(
        key="mlops",
        name="The MLOps Pragmatist ⚙️",
        role="Infrastructure and cost realist",
        system_prompt=(
            "Evaluate solo-developer feasibility. You care about PyTorch FSDP, Docker, GPU cost on RunPod. "
            "Kill ideas needing a 100-GPU cluster. Always estimate: 'this would cost ~$X/month to run.'"
        ),
    ),
    "domain_expert": Agent(
        key="domain_expert",
        name="The Domain Expert 🎯",
        role="Real-world niche validator",
        system_prompt=(
            "Ground AI in specific domains: HRV/VO2max/lactate tracking, ETF tax-loss harvesting, "
            "theorem proving, computational biology. Reject generic 'AI for X' without deep domain specificity. "
            "Always name a concrete target user who would pay TODAY."
        ),
    ),
    "finance": Agent(
        key="finance",
        name="The Monetization Expert 💰",
        role="Revenue model strategist",
        system_prompt=(
            "Ask 'Who writes the first check, and how much?' every time. "
            "Push for B2B SaaS or API-as-a-service with LTV > $1000. "
            "Reject consumer apps needing viral growth. Always propose a specific pricing tier."
        ),
    ),
}

moderator = Agent(
    key="moderator",
    name="The CEO Moderator 👑",
    role="Executive decision maker",
    system_prompt=(
        "You are the CEO. Read the transcript and the final evolved idea pool. "
        "Select the single best idea that survived the evolutionary pressure. "
        "Explain WHY it survived (technical moat + monetizability + solo-founder feasibility). "
        "Give one concrete, specific next step the founder should take THIS WEEK."
    ),
)


# ══════════════════════════════════════════════════════════════
#  EVOLUTIONARY EVALUATOR
# ══════════════════════════════════════════════════════════════

EVALUATOR_SYSTEM = (
    "You are an evolutionary selection engine for startup ideas. "
    "Score ideas based on the arguments made in the debate. "
    "KEEP ideas with score ≥7, MUTATE ideas with score 4–6 (fix the flaw raised in debate), "
    "DROP ideas with score ≤3 and replace with a crossover of the two best ideas. "
    "Evolved ideas must be MORE technically specific and monetizable than their predecessors."
)

def run_evolution(idea_pool: list[str], debate_transcript: str, generation: int) -> list[str]:
    section(f"⚗️  EVOLUTIONARY PRESSURE — After Round {generation}")

    prompt = (
        f"Current idea pool:\n{json.dumps(idea_pool, indent=2)}\n\n"
        f"What was argued in the debate (last 3000 chars):\n{debate_transcript[-3000:]}\n\n"
        f"Apply evolutionary selection. Return exactly {IDEA_POOL_SIZE} evolved ideas."
    )

    try:
        res  = call_api(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=EVALUATOR_SYSTEM,
                temperature=0.4,
                response_mime_type="application/json",
                response_schema=EvolutionResult,
            ),
        )
        data    = json.loads(res.text)
        scores  = data.get("scores", [])
        evolved = data.get("evolved_ideas", idea_pool)

        for s in scores:
            score_val   = s.get("score", 0)
            verdict_str = s.get("verdict", "")
            idx         = s.get("idea_index", 0)
            idea_preview= idea_pool[idx][:65] if idx < len(idea_pool) else "?"
            bar         = "█" * score_val + "░" * (10 - score_val)
            vc = "\033[92m" if "KEEP" in verdict_str.upper() else \
                 "\033[93m" if "MUTATE" in verdict_str.upper() else "\033[91m"
            print(f"  [{bar}] {score_val}/10  {vc}{verdict_str[:45]}{RESET}")
            print(f"  {DIM}↳ {idea_preview}…{RESET}\n")

        evolved = (evolved + idea_pool)[:IDEA_POOL_SIZE]  # safety pad

        print(f"\n  {BOLD}Evolved pool:{RESET}")
        for i, idea in enumerate(evolved, 1):
            print(f"  {BOLD}#{i}{RESET} {idea[:110]}")

        log(f"\n{'='*60}\nEVOLUTION — Generation {generation}")
        for i, idea in enumerate(evolved, 1):
            log(f"  #{i} {idea}")

        return evolved

    except Exception as e:
        print(f"  ⚠️  Evolution failed ({str(e)[:60]}), keeping current pool.")
        return idea_pool


# ══════════════════════════════════════════════════════════════
#  SEED GENERATION
# ══════════════════════════════════════════════════════════════

HARDCODED_SEEDS = [
    "A Mamba-SSM fine-tuned on HRV/VO2max wearable data to predict athlete recovery windows — B2B API for coaching platforms.",
    "LLM-powered ETF tax-loss harvesting engine that optimises wash-sale rules across multi-account portfolios — B2B SaaS for RIAs.",
    "JEPA-based self-supervised model for formal theorem proving assistance, targeting quant researchers writing in Lean4.",
    "Edge-deployable transformer (<50M params) for real-time ECG anomaly detection — licensed to wearable OEMs.",
]

def generate_seed_ideas() -> list[str]:
    section("🌱  SEEDING INITIAL IDEA POOL")
    print("  Generating diverse deep-tech startup seeds…\n")

    prompt = (
        f"Generate exactly {IDEA_POOL_SIZE} deep-tech startup ideas for a solo AI researcher.\n"
        "Domains: physiological AI, financial AI, novel model architectures, domain-specific LLMs.\n"
        f"Return ONLY a JSON array of {IDEA_POOL_SIZE} strings (1–2 sentences each). No preamble."
    )
    try:
        res  = call_api(model=MODEL, contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.9))
        raw  = res.text.strip().lstrip("```json").rstrip("```").strip()
        ideas = json.loads(raw)
        if isinstance(ideas, list) and len(ideas) >= IDEA_POOL_SIZE:
            ideas = [str(i) for i in ideas[:IDEA_POOL_SIZE]]
            for i, idea in enumerate(ideas, 1):
                print(f"  #{i} {idea[:100]}")
            log("SEED IDEAS:\n" + "\n".join(f"  #{i+1} {idea}" for i, idea in enumerate(ideas)))
            return ideas
    except Exception as e:
        print(f"  ⚠️  Seed generation error ({e}). Using hardcoded seeds.")

    for i, idea in enumerate(HARDCODED_SEEDS, 1):
        print(f"  #{i} {idea[:100]}")
    return HARDCODED_SEEDS


# ══════════════════════════════════════════════════════════════
#  DEBATE ROUND
# ══════════════════════════════════════════════════════════════

def run_debate_round(idea_pool: list[str], generation: int) -> list[str]:
    section(f"🎙️  BOARDROOM DEBATE — Round {generation + 1} / {GENERATIONS}")

    ideas_str = "\n".join(f"  {i+1}. {idea}" for i, idea in enumerate(idea_pool))
    opening   = (
        f"[Moderator]: Round {generation + 1} — ideas on the table:\n{ideas_str}\n\n"
        "Pressure-test these. Visionary, lead us in."
    )
    print(f"  {clr('moderator', opening)}\n")
    log(f"\n{'='*60}\nDEBATE ROUND {generation + 1}\n{'='*60}\n{opening}")

    history        = [opening]
    valid_keys     = list(agents.keys())
    next_agent_key = "visionary"

    for turn in range(TURNS_PER_ROUND):
        agent = agents[next_agent_key]
        print(f"  {DIM}[Turn {turn+1}/{TURNS_PER_ROUND} — {agent.name}]{RESET}")

        recent = "\n\n".join(history[-CONTEXT_WINDOW:])
        reply, next_key = agent.speak(recent, valid_keys)

        formatted = f"[{agent.name}]: {reply}"
        print(f"  {clr(agent.key, formatted)}\n")
        log(formatted + "\n" + "-"*50)

        history.append(formatted)
        next_agent_key = next_key

    return history


# ══════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ══════════════════════════════════════════════════════════════

def run_final_verdict(idea_pool: list[str], all_history: list[str]):
    section("👑  CEO FINAL VERDICT")
    ideas_str = "\n".join(f"  {i+1}. {idea}" for i, idea in enumerate(idea_pool))
    prompt    = (
        f"The evolutionary boardroom debate is over.\n\n"
        f"Final idea pool:\n{ideas_str}\n\n"
        f"Recent debate (last 15 messages):\n"
        f"{chr(10).join(all_history[-15:])}\n\n"
        "Give your final executive verdict."
    )
    try:
        res = call_api(
            model=MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=moderator.system_instruction,
                temperature=0.4,
            ),
        )
        verdict = res.text
        print(f"  {clr('moderator', verdict)}")
        log(f"\n{'='*60}\nCEO FINAL VERDICT\n{'='*60}\n{verdict}")
    except Exception as e:
        print(f"  ⚠️  Verdict error: {e}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    tier_label = "Paid (fast)" if PAID_TIER else "Free (throttled)"

    # Total calls = 1 seed + GENERATIONS*TURNS_PER_ROUND debate
    #             + (GENERATIONS-1) evolution + 1 verdict
    total_calls = 1 + GENERATIONS * TURNS_PER_ROUND + (GENERATIONS - 1) + 1
    est_minutes = (total_calls * CALL_DELAY) / 60

    print(f"""
{BOLD}╔══════════════════════════════════════════════════════════════╗
║          BOARDROOM EVOLUTION ENGINE  v1.1                    ║
║   Multi-Agent Debate  ×  Evolutionary Selection              ║
╚══════════════════════════════════════════════════════════════╝{RESET}
  Generations   : {GENERATIONS}
  Agents        : {len(agents)}
  Turns / Round : {TURNS_PER_ROUND}
  Idea Pool     : {IDEA_POOL_SIZE}
  API Tier      : {tier_label}  (delay = {CALL_DELAY}s / call)
  Total calls   : ~{total_calls}
  Est. runtime  : ~{est_minutes:.1f} min
  Log File      : {LOG_FILE}
""")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"BOARDROOM EVOLUTION ENGINE v1.1 — {TIMESTAMP}\n\n")

    idea_pool   = generate_seed_ideas()
    all_history = []

    for gen in range(GENERATIONS):
        round_history = run_debate_round(idea_pool, gen)
        all_history.extend(round_history)

        if gen < GENERATIONS - 1:
            idea_pool = run_evolution(idea_pool, "\n".join(round_history), gen + 1)

    run_final_verdict(idea_pool, all_history)

    print(f"\n{BOLD}✔  Session complete → {LOG_FILE}{RESET}\n")


if __name__ == "__main__":
    main()