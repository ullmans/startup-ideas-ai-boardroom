import os
import time
import json
import itertools
from datetime import datetime
from xmlrpc import client
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, RetryError
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. API & Configuration Setup
# ==========================================
keys_str = os.getenv("GEMINI_API_KEYS", "")
api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]

key_pool = itertools.cycle(api_keys)

def get_next_client():
    next_key = next(key_pool)
    if not next_key:
        print("Error: API key not found.")
        exit(1)
    return genai.Client(api_key=next_key)

MODEL_NAME = "gemini-2.5-flash"

# ==========================================
# 2. Schema Definitions for Structured Output
# ==========================================
class AgentResponse(BaseModel):
    response: str = Field(description="Your spoken response in the boardroom.")
    next_speaker: str = Field(description="The exact key of the agent who should speak next to keep the debate productive.")

class Idea(BaseModel):
    title: str = Field(description="A short, catchy title for the startup idea.")
    description: str = Field(description="A clear description of the product, the target audience, and the problem it solves.")

class IdeaList(BaseModel):
    ideas: list[Idea] = Field(description="List of exactly 4 top ideas that survived the debate.")

class TopIdeas(BaseModel):
    ideas: list[Idea] = Field(description="List of exactly 3 top ideas selected for final debate.")

# ==========================================
# 3. Robust API Calls with Tenacity (Optimized for 429)
# ==========================================
# הגדרה מורחבת של Tenacity שתדע להתמודד בצורה אגרסיבית יותר עם חריגה מהמכסה
@retry(
    wait=wait_exponential(multiplier=4, min=10, max=120), 
    stop=stop_after_attempt(6),
    reraise=True
)
def call_gemini_structured(client, prompt, system_instruction, schema):
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=schema,
        )
    )

@retry(
    wait=wait_exponential(multiplier=4, min=10, max=120), 
    stop=stop_after_attempt(6),
    reraise=True
)
def call_gemini_text(client, prompt, system_instruction, temperature=0.7):
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
    )

def generate_discussion_summary(client, message_history):
    if len(message_history) <= 1:
        return "The discussion has just begun. No ideas have been proposed yet."
        
    full_transcript = "\n\n".join(message_history)
    prompt = f"Review the following boardroom transcript and list out the raw startup ideas pitched so far as bullet points. Do not include critiques or analysis, just the core ideas:\n\n{full_transcript}"
    
    system_instruction = (
        "You are an expert executive scribe. Provide a clean, bulleted list of the unique startup ideas proposed. "
        "Keep descriptions brief and technical. Do not genericize."
    )
    
    try:
        response = call_gemini_text(client, prompt, system_instruction, temperature=0.2)
        return response.text
    except Exception as e:
        print(f"[Warning] Failed to generate summary (429 or connection): {e}")
        return "Summary temporarily unavailable. Continue pitching new ideas."

# ==========================================
# 4. Agent Class Definition
# ==========================================
class Agent:
    def __init__(self, key, name, role, system_prompt):
        self.key = key
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.system_instruction = (
            f"You are participating in Phase 1 of a live startup boardroom session. "
            f"Your role: {role}.\nInstructions: {system_prompt}\n"
            f"CRITICAL RULES FOR PHASE 1:\n"
            f"1. DO NOT analyze, criticize, or dive deep into previous ideas. Absolutely NO pros and cons layout yet!\n"
            f"2. Your ONLY goal in this phase is to pitch NEW, diverse, and creative startup ideas from the perspective of your role, "
            f"or pivot the conversation to entirely new industries to ensure a wide variety of options.\n"
            f"3. Keep your response short (2-3 sentences), pitching one clear product concept. Always answer in English."
        )

    def speak(self, client, structured_history, valid_keys, phase_prompt="Pitch a NEW startup idea or pivot to a new domain. Do not critique current ideas."):
        prompt = (
            f"{structured_history}\n\n"
            f"It is your turn to speak. {phase_prompt} Select the next speaker from this list: {valid_keys}. "
            f"Do not select yourself."
        )

        # one try at a time with aggressive retries and key rotation in case of rate limits or other errors:
        def one_try_of_speaking(current_client):   
            response = call_gemini_structured(current_client, prompt, self.system_instruction, AgentResponse)
            
            if not response.text:
                 return "(Silent - Empty response received)", valid_keys[0]

            data = json.loads(response.text)
            reply_text = data.get("response", "(Silent)")
            next_speaker = data.get("next_speaker", "").lower().strip()
            
            if next_speaker not in valid_keys or next_speaker == self.key:
                current_idx = valid_keys.index(self.key)
                next_speaker = valid_keys[(current_idx + 1) % len(valid_keys)]
                
            return reply_text, next_speaker

        last_error = None
        current_client = client
        for attempt in range(3):
            try:
                return one_try_of_speaking(current_client)
            except Exception as e:
                last_error = e
                if attempt < 2: # only get next client if we have more tries
                    current_client = get_next_client() # החלפת מפתח במקרה של תקלה

        print(f"\n[Error] Agent {self.name} encountered an issue: {last_error}. Passing turn.")
        current_idx = valid_keys.index(self.key)
        return "(Remained silent to protect rate limits)", valid_keys[(current_idx + 1) % len(valid_keys)]

# ==========================================
# 5. Agent Team Initialization
# ==========================================
agents = {
    "visionary": Agent(
        key="visionary", name="The Product Visionary", role="The ideas person",
        system_prompt="Pitch an innovative, high-value Applied AI product that solves a massive user pain point. Keep it focused on intuitive UX/product value."
    ),
    "applied_ai": Agent(
        key="applied_ai", name="The Applied AI Engineer", role="Practical neural network implementer",
        system_prompt="Pitch a creative product idea centered around RAG, fine-tuning small models, embeddings, or CV. Do not review code, just pitch the concept."
    ),
    "mlops": Agent(
        key="mlops", name="The Solo-Dev MLOps", role="Feasibility expert",
        system_prompt="Pitch a product concept that leverages lightweight architectures, serverless tech, or automation tools. What is a highly profitable microservice a solo dev can build?"
    ),
    "domain_expert": Agent(
        key="domain_expert", name="The Niche Domain Expert", role="Industry explorer",
        system_prompt="Forcibly pivot the session to unexpected industries (e.g., agriculture, legal tech, logistics, manufacturing). Pitch an AI concept for a niche field."
    ),
    "critic": Agent(
        key="critic", name="The Ruthless Critic", role="The Guardrail",
        system_prompt="In this phase, instead of attacking, pitch an alternative idea that bypasses common traps (like heavy tech-giant competition or high customer acquisition costs)."
    ),
    "finance": Agent(
        key="finance", name="The Go-To-Market Strategist", role="Business and monetization expert",
        system_prompt="Pitch a high-margin B2B SaaS or professional tool concept where businesses will happily pay from day one. Focus on direct utility."
    )
}

moderator = Agent(
    key="moderator", name="The Moderator", role="CEO",
    system_prompt="You are the CEO. Ensure the team stays broad, does not drill down, and collects a wide net of unique startup ideas."
)

INJECTED_IDEAS = [
    {
        "title": "MLOps DevTool for Distributed Training",
        "description": "A platform or container-based CLI for small/medium research teams. It analyzes their code and automatically builds an optimal distributed training environment (managing Multi-GPU, FSDP, Tensor Parallelism, Mamba architectures) to save configuration time and expensive cloud compute hours."
    },
    {
        "title": "AI Real Estate Forecasting for the Periphery",
        "description": "An AI model predicting future property prices for single-family homes in developing/peripheral cities (e.g., Yeruham, Ofakim). It uses demographic data, master plans, and infrastructure developments to identify undervalued assets before the broader market surges."
    },
    {
        "title": "Tax Optimization Platform for Long-Term DIY Investors",
        "description": "An analytical tool connecting to brokerage accounts for private investors managing global ETF portfolios (e.g., S&P 500). It calculates long-term mathematical tax optimization, ideal rebalancing timing, and handles the tax differences between accumulating vs. distributing funds."
    }
]

# ==========================================
# 6. The Orchestrator Engine
# ==========================================
def run_dynamic_boardroom():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"boardroom_session_{timestamp}.txt"
    ideas_filename = f"top_ideas_{timestamp}.json"
    
    print(f"=== Phase 1: Broad Idea Generation ===")
    print(f"[System] Writing session live to: {log_filename}\n")
    
    initial_prompt = (
        "[The Moderator]: Welcome team. We are in PHASE 1: WIDE BRAINSTORMING. "
        "Our goal is to build a massive list of completely different Applied AI startup ideas for a solo developer. "
        "CRITICAL RULE: DO NOT analyze or criticize any ideas yet! No deep-dives allowed. "
        "Just throw unique product concepts on the table. Visionary, pitch the first idea to get us rolling."
    )
        
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("=== PHASE 1: Wide Brainstorming Transcript ===\n\n")
        f.write(initial_prompt + "\n")
    
    max_turns =  20 # 6 תורות מספיקים כדי שכל סוכן יזרוק רעיון רוחבי אחד מבלי להתקבע
    message_history = [initial_prompt]
    valid_agent_keys = list(agents.keys())
    next_agent_key = "visionary"
    
    # --- PHASE 1: THE WIDE PITCHES ---
    for turn in range(max_turns):
        current_agent = agents[next_agent_key]
        print(f"[System] Turn {turn + 1}/{max_turns}: {current_agent.name} is pitching a new concept...")
        
        summary_client = get_next_client()
        discussion_summary = generate_discussion_summary(summary_client, message_history)
        
        immediate_context = "\n\n".join(message_history[-1:]) # רק התגובה האחרונה כדי שלא ייגררו לניתוח
        structured_history = (
            f"### STARTUP IDEAS PROPOSED SO FAR:\n{discussion_summary}\n\n"
            f"### LAST PITCH:\n{immediate_context}"
        )
        
        agent_client = get_next_client()
        reply_text, chosen_next_speaker = current_agent.speak(agent_client, structured_history, valid_agent_keys)
        
        formatted_reply = f"[{current_agent.name}]: {reply_text}"
        message_history.append(formatted_reply)
        
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"\n{formatted_reply}\n")
            f.write("-" * 50 + "\n")
        
        next_agent_key = chosen_next_speaker
        time.sleep(6) # הגדלת ההשהיה ל-6 שניות למניעת 429 כבר בשלב הראשון
        
    print("\n[System] Phase 1 complete. Extracting 4 ideas from the pool...")
    
    # --- PHASE 1 EXTRACTION ---
    extract_client = get_next_client()
    full_transcript = "\n\n".join(message_history)
    extract_prompt = f"The brainstorming phase has concluded. Here is the transcript of raw pitches:\n{full_transcript}\n\nAs the Moderator, filter and select EXACTLY the 4 most unique and interesting startup ideas from the text."
    
    try:
        extraction_response = call_gemini_structured(extract_client, extract_prompt, moderator.system_instruction, IdeaList)
        extracted_ideas = json.loads(extraction_response.text)
        
        with open(ideas_filename, "w", encoding="utf-8") as f:
            json.dump(extracted_ideas, f, indent=4, ensure_ascii=False)
            
        print(f"\n[System] ✔ Top 4 ideas saved to '{ideas_filename}'.")
    except Exception as e:
        print(f"[Error] Failed to extract ideas due to rate limits/error: {e}")
        return

    # --- USER APPROVAL ---
    print("\n" + "="*50)
    print("⏸️  USER APPROVAL REQUIRED")
    print(f"Please open '{ideas_filename}', review/edit the 4 ideas, and save.")
    print("Your 3 predefined ideas (MLOps, Real Estate, Tax) will be injected next.")
    print("="*50)
    
    while True:
        input("\nPress ENTER when you are ready to continue to Phase 2 (Deep Dive & Critique)...")
        try:
            with open(ideas_filename, "r", encoding="utf-8") as f:
                approved_ideas_data = json.load(f)
            approved_ideas_list = approved_ideas_data.get("ideas", [])
            break
        except json.JSONDecodeError:
            print("[Error] Invalid JSON format. Please fix and try again.")
            
    # --- PHASE 2: INJECTION & DEEP DIVE ---
    print("\n=== Phase 2: Idea Injection & Deep Dive ===")
    all_ideas_to_evaluate = approved_ideas_list + INJECTED_IDEAS
    deep_dive_log = []
    
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write("\n\n" + "="*50 + "\n")
        f.write("=== PHASE 2: DEEP DIVE & CRITIQUE (PROS & CONS) ===\n")
        f.write("="*50 + "\n")

    for idx, idea in enumerate(all_ideas_to_evaluate, 1):
        idea_title = idea.get("title", f"Idea {idx}")
        idea_desc = idea.get("description", "")
        
        print(f"\n[Deep Dive - Idea {idx}/{len(all_ideas_to_evaluate)}]: {idea_title}")
        idea_header = f"\n\n--- Analysis of Idea {idx}: {idea_title} ---\nDescription: {idea_desc}\n"
        deep_dive_log.append(idea_header)
        
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(idea_header)

        # מעבר מסודר בין הסוכנים לניתוח פרקטי ליזם בודד
        for agent_key, agent in agents.items():
            eval_prompt = (
                f"We are conducting a deep dive on the following startup idea:\n"
                f"Title: {idea_title}\n"
                f"Description: {idea_desc}\n\n"
                f"Evaluate this idea strictly from your role's perspective ({agent.role}) for a SOLO DEVELOPER. "
                f"What are the PROS? What are the CONS? Is it practically buildable and maintainable by one person? "
                f"Keep your response under 4 sentences, packed with analytical value."
            )
            
            eval_text = None
            last_error = None
            for attempt in range(3):
                try:
                    eval_client = get_next_client()
                    response = call_gemini_text(eval_client, eval_prompt, agent.system_instruction)
                    eval_text = response.text.strip()
                    break
                except Exception as e:
                    last_error = e
                    print(f"  [Rate Limit Warning] Attempt {attempt+1}/3 failed for agent {agent.name}: {e}")
                    if attempt < 2:
                        time.sleep(15) # במקרה של תקלה, נמתין 15 שניות שלמות כדי לתת ל-API להתקרר
            
            if eval_text is None:
                eval_text = f"(Evaluation skipped due to persistent rate limits. Last error: {last_error})"
                
            formatted_eval = f"[{agent.name}]: {eval_text}\n"
            deep_dive_log.append(formatted_eval)
            print(f"  ↳ {agent.name} completed evaluation.")
            
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(formatted_eval + "\n")
                f.write("-" * 50 + "\n")

            # ההשהיה הקריטית: 8 שניות שלמות בין סוכן לסוכן כדי למנוע לחלוטין את שגיאה 429
            time.sleep(8) 
            
    # --- PHASE 3: OPEN DEBATE ON TOP 3 IDEAS ---
    print("\n=== Phase 3: Selection & Open Debate on Top 3 Ideas ===")
    print("[System] The Moderator is selecting the Top 3 ideas from the Deep Dive...")
    
    selection_prompt = (
        f"Here is the complete analysis and critique from Phase 2:\n"
        f"{''.join(deep_dive_log)}\n\n"
        f"As the CEO, review the feedback and select EXACTLY the 3 most promising ideas "
        f"for a final open debate. Provide their titles and descriptions."
    )
    
    top_3_ideas = []
    for attempt in range(3):
        try:
            sel_client = get_next_client()
            sel_response = call_gemini_structured(sel_client, selection_prompt, moderator.system_instruction, TopIdeas)
            data = json.loads(sel_response.text)
            top_3_ideas = data.get("ideas", [])
            if len(top_3_ideas) > 0:
                break
        except Exception as e:
            print(f"  [Warning] Attempt {attempt+1}/3 failed to extract top 3 ideas: {e}")
            if attempt < 2:
                time.sleep(15)
                
    if not top_3_ideas:
        print("[Warning] Failed to select Top 3 ideas. Falling back to the first 3 ideas evaluated.")
        top_3_ideas = all_ideas_to_evaluate[:3]

    with open(log_filename, "a", encoding="utf-8") as f:
        f.write("\n\n" + "="*50 + "\n")
        f.write("=== PHASE 3: OPEN DEBATE ON TOP 3 IDEAS ===\n")
        f.write("="*50 + "\n")

    # Update Agent system instructions for Debate
    for agent in agents.values():
        agent.system_instruction = (
            f"You are participating in Phase 3 of a live startup boardroom session. "
            f"Your role: {agent.role}.\nInstructions: {agent.system_prompt}\n"
            f"CRITICAL RULES FOR PHASE 3:\n"
            f"1. We are now DEBATING the top 3 selected ideas. You MUST analyze, criticize, and argue for/against them.\n"
            f"2. React directly to the previous speakers. Build on their points or challenge them based on your role.\n"
            f"3. Keep your response sharp and concise (3-4 sentences). Always answer in English."
        )

    intro = "[The Moderator]: Welcome to Phase 3. Based on the deep dives, I've selected the Top 3 ideas for us to debate openly:\n"
    for i, idea in enumerate(top_3_ideas, 1):
        title = idea.get('title', f"Idea {i}") if isinstance(idea, dict) else getattr(idea, 'title', f"Idea {i}")
        desc = idea.get('description', "") if isinstance(idea, dict) else getattr(idea, 'description', "")
        intro += f"{i}. {title}: {desc}\n"
    intro += "\nWe need to debate these and figure out which ONE is the absolute best for a solo developer. Critic, start us off."

    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(intro + "\n")
        f.write("-" * 50 + "\n")

    debate_history = [intro]
    debate_log = [intro]
    next_agent_key = "critic"
    debate_turns = 100
    
    for turn in range(debate_turns):
        current_agent = agents[next_agent_key]
        print(f"[System] Phase 3 - Turn {turn + 1}/{debate_turns}: {current_agent.name} is speaking...")
        
        immediate_context = "\n\n".join(debate_history[-3:])
        structured_history = f"### RECENT DEBATE CONTEXT:\n{immediate_context}"
        
        agent_client = get_next_client()
        phase3_prompt = "React to the recent discussion. Defend your favorite idea or ruthlessly critique the others."
        reply_text, chosen_next_speaker = current_agent.speak(agent_client, structured_history, valid_agent_keys, phase_prompt=phase3_prompt)
        
        formatted_reply = f"[{current_agent.name}]: {reply_text}"
        debate_history.append(formatted_reply)
        debate_log.append(formatted_reply)
        
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"\n{formatted_reply}\n")
            f.write("-" * 50 + "\n")
        
        next_agent_key = chosen_next_speaker
        time.sleep(8)

    # --- PHASE 4: FINAL WEIGHTED RANKING ---
    print("\n=== Phase 4: Final Verdict ===")
    print("[System] The Moderator is calculating the final scores...")
    
    full_evaluations_text = "--- PHASE 2 DEEP DIVES ---\n" + "".join(deep_dive_log) + "\n\n--- PHASE 3 DEBATE ---\n" + "\n\n".join(debate_log)
    
    ranking_prompt = (
        f"Here is the complete analysis and critique from all agents for the startup ideas:\n"
        f"{full_evaluations_text}\n\n"
        f"As the CEO and Moderator, review all the feedback carefully. "
        f"Provide a final, weighted ranking of ALL the ideas evaluated, explicitly looking at the constraints of a solo developer. "
        f"Present the ranking clearly, declare the single absolute WINNING idea, and outline a concrete 'Next Step' for execution."
    )
    
    verdict_text = None
    last_error = None
    for attempt in range(3):
        try:
            final_client = get_next_client()
            final_response = call_gemini_text(final_client, ranking_prompt, moderator.system_instruction, temperature=0.5)
            verdict_text = final_response.text
            break
        except Exception as e:
            last_error = e
            print(f"  [Warning] Attempt {attempt+1}/3 failed for Moderator ranking: {e}")
            if attempt < 2:
                time.sleep(15)

    if verdict_text is None:
        verdict_text = f"Moderator failed to compile ranking due to technical error: {last_error}"
        
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write("\n\n" + "="*50 + "\n")
        f.write("=== FINAL VERDICT & RANKING ===\n")
        f.write("="*50 + "\n")
        f.write(verdict_text)
        
    print(f"\n[System] ✔ Session completely finished. Full transcript and final ranking saved to: {log_filename}")

if __name__ == "__main__":
    run_dynamic_boardroom()