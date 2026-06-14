import os
import time
import json
import itertools
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. API & Configuration Setup
# ==========================================

# רשימת מפתחות לרוטציה
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
# 2. Schema Definition for Structured Output
# ==========================================
class AgentResponse(BaseModel):
    response: str = Field(description="Your spoken response in the boardroom.")
    next_speaker: str = Field(description="The exact key of the agent who should speak next to keep the debate productive.")

# ==========================================
# 3. Robust API Calls with Tenacity
# ==========================================
@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5))
def call_gemini_safe(client, prompt, system_instruction):
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.8,
            response_mime_type="application/json",
            response_schema=AgentResponse,
        )
    )

@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5))
def generate_discussion_summary(client, message_history):
    """מריץ פנייה מהירה ל-Gemini כדי לקבל תמצות ממוקד של כל הדיון עד כה"""
    if len(message_history) <= 1:
        return "The discussion has just begun. No arguments have been made yet."
        
    full_transcript = "\n\n".join(message_history)
    prompt = f"Review the following boardroom discussion transcript and provide a highly concise, technical bullet-point summary of the core ideas, critiques, and constraints raised so far:\n\n{full_transcript}"
    
    system_instruction = (
        "You are an expert executive scribe. Provide a sharp, dense summary of the debate. "
        "Preserve technical jargon (like FSDP, Mamba, HRV, ETFs) and specific constraints mentioned by the experts. Do not genericize."
    )
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3 # טמפרטורה נמוכה לסיכום עובדתי ומדויק
        )
    )
    return response.text

# ==========================================
# 4. Agent Class Definition
# ==========================================
class Agent:
    def __init__(self, key, name, role, system_prompt):
        self.key = key
        self.name = name
        self.role = role
        self.system_instruction = (
            f"You are participating in a live startup brainstorming boardroom session. "
            f"Your role: {role}.\nInstructions: {system_prompt}\n"
            f"IMPORTANT: You are provided with an executive summary of the debate so far, plus the immediate context. "
            f"React directly to the recent speakers, build on ideas, or disagree based on your expert perspective. "
            f"Keep your responses concise, sharp, and impactful. Always answer in English."
        )

    def speak(self, client, structured_history, valid_keys):
        prompt = (
            f"{structured_history}\n\n"
            f"It is your turn to speak. Provide your response, and then intelligently select the next speaker from this list: {valid_keys}. "
            f"Do not select yourself."
        )

        try:
            # 1. קריאה ל-API
            response = call_gemini_safe(client, prompt, self.system_instruction)
            
            # 2. וידוא שה-API בכלל החזיר טקסט לפני שמנסים לנתח אותו כ-JSON
            if not response.text:
                 print(f"\n[Warning] {self.name} received an empty response from the API (possible safety block).")
                 return "(Silent - Empty response received)", valid_keys[0]

            # 3. ניתוח ה-JSON
            try:
                data = json.loads(response.text)
            except json.JSONDecodeError as json_err:
                 print(f"\n[Warning] {self.name} failed to parse JSON. Raw API response was:\n{response.text}\nJSON Error: {json_err}")
                 return "(Silent - Failed to parse AI output)", valid_keys[0]

            reply_text = data.get("response", "(Silent)")
            next_speaker = data.get("next_speaker", "").lower().strip()
            
            if next_speaker not in valid_keys or next_speaker == self.key:
                current_idx = valid_keys.index(self.key)
                next_speaker = valid_keys[(current_idx + 1) % len(valid_keys)]
                
            return reply_text, next_speaker

        except RetryError as retry_err:
            real_error = retry_err.last_attempt.exception()
            print(f"\n[API Block Error] {self.name} API request failed. Details: {real_error}")
            return f"(Silent due to API connection drop)", valid_keys[0]
            
        except Exception as e:
            # כאן אנחנו מדפיסים את השגיאה במלואה כדי לראות מה היא!
            import traceback
            print(f"\n[Unexpected Error] Agent {self.name} crashed.")
            print(traceback.format_exc()) # ידפיס בדיוק באיזו שורה ולמה קרתה השגיאה
            return f"(Remained silent due to critical error)", valid_keys[0]

# ==========================================
# 5. Agent Team Initialization
# ==========================================
agents = {
    "visionary": Agent(
        key="visionary",
        name="The Product Visionary",
        role="The ideas person who focuses on solving painful user problems with AI",
        system_prompt="Push for products that solve massive pain points. Propose intuitive user experiences. If an idea is too technical, ask 'How does this actually help the end user?'."
    ),
    "applied_ai": Agent(
        key="applied_ai",
        name="The Applied AI Engineer",
        role="Practical neural network implementer",
        system_prompt="Focus on applied neural networks. Push for architectures using RAG (Retrieval-Augmented Generation), fine-tuning small open-source models (like Llama/Mistral), embeddings, or applied Computer Vision. Reject ideas that require training foundational models from scratch."
    ),
    "mlops": Agent(
        key="mlops",
        name="The Solo-Dev MLOps",
        role="Evaluates deployment feasibility and costs",
        system_prompt="You are obsessed with low-cost, high-speed deployment for a solo developer. Push for Serverless GPUs, Dockerized microservices, and efficient PyTorch inference. Kill ideas that require expensive multi-GPU training clusters or massive data pipelines."
    ),
    "domain_expert": Agent(
            key="domain_expert",
            name="The Niche Domain Expert",
            role="Explores various high-value, data-rich industries",
            system_prompt="Your goal is to explore multiple, completely different B2B or prosumer domains (e.g., legal tech, agriculture, supply chain, niche manufacturing, specialized finance, local municipal services). If the team gets stuck on one industry for too long, forcibly pivot the conversation to a new, unexpected domain. Avoid sports or fitness."
        ),
    "critic": Agent(
        key="critic",
        name="The Ruthless Critic",
        role="The Devil's Advocate",
        system_prompt="Attack the ideas presented. Look for high customer acquisition costs (CAC), tech giant threats (e.g., 'OpenAI will just build this next week'), and weak competitive moats. Never completely agree with the previous speaker."
    ),
    "finance": Agent(
        key="finance",
        name="The Go-To-Market Strategist",
        role="Business and monetization expert",
        system_prompt="Keep the team grounded on revenue. Ask 'Who is paying for this?' and 'How do we get the first 10 paying customers without a marketing budget?'. Strongly favor B2B SaaS, professional internal tools, or high-ticket B2C over free consumer apps."
    )
}

moderator = Agent(
    key="moderator",
    name="The Moderator",
    role="Summarizes the boardroom session and makes executive decisions",
    system_prompt="You are the CEO. Read the final summary of the boardroom. Extract the absolute best startup idea that survived the debate. Provide a concrete, technical 'Next Step' for the solo founder."
)

# ==========================================
# 6. The Orchestrator Engine
# ==========================================
def run_dynamic_boardroom():
    boardroom_dir = "boardrooms"
    os.makedirs(boardroom_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(boardroom_dir, f"boardroom_session_{timestamp}.txt")
    
    print(f"=== Opening the Boardroom Doors ===")
    print(f"[System] Writing session live to: {log_filename}\n")
    
    initial_prompt = (
            "[The Moderator]: Welcome everyone. We are here to find a highly profitable, Applied AI startup idea for a solo developer. "
            "Today, we are casting a wide net. I DO NOT want us locking into a specific industry in the first few minutes. "
            "We need to explore at least 3 or 4 completely different, unexpected B2B or high-ticket niches before we decide which one has the best potential to drill down into. "
            "Visionary, start us off by pitching an unconventional AI product in an industry people usually ignore."
        )
        
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("=== Boardroom Brainstorming Transcript ===\n\n")
        f.write(initial_prompt + "\n")
    
    max_turns = 15
    message_history = [initial_prompt]
    valid_agent_keys = list(agents.keys())
    next_agent_key = "visionary"
    
    for turn in range(max_turns):
        current_agent = agents[next_agent_key]
        print(f"[System] Turn {turn + 1}/{max_turns}: Preparing context for {current_agent.name}...")
        
        # צעד א': הפקת תמצות של כל ההיסטוריה עד כה באמצעות לקוח מהרוטציה
        try:
            summary_client = get_next_client()
            discussion_summary = generate_discussion_summary(summary_client, message_history)
        except Exception as e:
            print(f"[Warning] Failed to generate summary: {e}")
            discussion_summary = "Summary unavailable due to error."
        
        # צעד ב': שליפת שתי התגובות האחרונות כלשונן לשמירה על רצף השיחה המיידי
        immediate_context = "\n\n".join(message_history[-2:])
        
        # צעד ג': הרכבת המבנה המשולב שישלח לסוכן
        structured_history = (
            f"### EXECUTIVE SUMMARY OF THE DEBATE SO FAR:\n{discussion_summary}\n\n"
            f"### IMMEDIATE CONTEXT (LAST REPLIES):\n{immediate_context}"
        )
        
        # צעד ד': הרצת הסוכן הדובר עם לקוח חדש מהרוטציה
        agent_client = get_next_client()
        reply_text, chosen_next_speaker = current_agent.speak(agent_client, structured_history, valid_agent_keys)
        
        formatted_reply = f"[{current_agent.name}]: {reply_text}"
        message_history.append(formatted_reply)
        
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"\n{formatted_reply}\n")
            f.write("-" * 50 + "\n")
        
        next_agent_key = chosen_next_speaker
        
        # השהיה קלה של 4 שניות בין תור לתור כדי להגן על ה-RPM (קצב הבקשות)
        time.sleep(4) 
        
    print("\n[System] Debate time limit reached. Calling the Moderator for the final verdict...")
    
    final_client = get_next_client()
    full_transcript = "\n\n".join(message_history)
    final_prompt = f"The debate has concluded. Here is the full transcript:\n{full_transcript}\n\nPlease provide your final executive summary and the winning idea."
    
    try:
        @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5))
        def get_final_verdict():
            return final_client.models.generate_content(
                model=MODEL_NAME,
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=moderator.system_instruction,
                    temperature=0.5
                )
            )
        final_response = get_final_verdict()
        verdict_text = final_response.text
    except Exception as e:
        verdict_text = f"Moderator failed to summarize due to error: {e}"
    
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write("\n\n=== Moderator's Final Verdict ===\n\n")
        f.write(verdict_text)
        
    print(f"[System] ✔ Session complete. Full transcript and verdict saved to: {log_filename}")

if __name__ == "__main__":
    run_dynamic_boardroom()