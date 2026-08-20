import os
import json

# Lazy initialization — the key is read from env at call time, NOT at import time.
# This ensures load_dotenv() in app.py has already been called.
_genai = None
_configured = False

def _ensure_configured():
    global _genai, _configured
    if _configured:
        return
    _configured = True
    try:
        import google.generativeai as genai
        _genai_module = genai
    except ImportError:
        print("WARNING: google-generativeai package not installed. AI features disabled.")
        return

    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        globals()['_genai'] = genai
        print("[OK] Gemini API configured successfully.")
    else:
        print("WARNING: GEMINI_API_KEY not found in environment. AI features will use fallback responses.")

def generate_health_tips(report_data):
    """
    Generate personalized health tips based on a patient's ML risk report.
    report_data should be a dict containing properties like age, bmi, tobacco, etc.
    """
    _ensure_configured()
    if not _genai:
        return "Regular hydration, a balanced diet, and daily 30-minute walks are recommended to maintain heart health. Please consult a doctor for advice."

    try:
        model = _genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        You are a highly qualified cardiac AI assistant. 
        Analyze the following patient data:
        - Age: {report_data.get('age')}
        - Sex: {'Male' if report_data.get('sex') == 1 else 'Female'}
        - Risk Level: {report_data.get('risk_level')}
        - Probability: {report_data.get('probability', 0) * 100:.1f}%
        - BMI: {report_data.get('bmi', 'Unknown')}
        - Tobacco History: {['No', 'Past', 'Current'][report_data.get('tobacco', 0)]}
        - Unhealthy Diet: {'Yes' if report_data.get('unhealthy_diet') == 1 else 'No'}
        - Stress Level: {report_data.get('stress_level', 'Unknown')}/10
        - Exercise: {report_data.get('exercise_freq', 'Unknown')} days/week
        
        Provide exactly 3 concise, actionable, and personalized health tips for this patient.
        Do not use markdown bolding (**) in the output. Keep each tip under 2 sentences.
        Output as a JSON array of strings. Address the patient directly as 'You'.
        """
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        tips_list = json.loads(text)
        return " | ".join(tips_list)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Stay hydrated, maintain a balanced diet, and monitor your vitals regularly."

def system_chatbot_prompt():
    """
    Returns the core system prompt for the 24/7 cardiac health chatbot.
    """
    return (
        "You are HeartGuard AI, a 24/7 cardiac health assistant. "
        "Your purpose is to answer general queries about heart health, "
        "cholesterol, blood pressure, exercises, and heart-healthy diets based on clinical guidelines. "
        "IMPORTANT RULES:\n"
        "1. You MUST NOT diagnose or prescribe medication.\n"
        "2. If the user asks about an emergency or severe symptoms (e.g., chest pain, numbness), "
        "immediately instruct them to call their local emergency number or use the SOS button on the dashboard.\n"
        "3. Keep answers empathetic, concise, and easy to understand.\n"
        "4. Always add a disclaimer that you are an AI and they should consult their specialized doctor "
        "(from the HeartGuard app) for a formal clinical plan."
    )

def handle_chat_message(user_message, conversation_history=[]):
    """
    Processes a chat message through the Gemini model and returns the response.
    """
    _ensure_configured()
    if not _genai:
        # Fallback: rule-based responses when no API key is set
        msg = user_message.lower()
        if any(w in msg for w in ['chest pain', 'emergency', 'heart attack', 'can\'t breathe']):
            return "This sounds like a medical emergency! Please call your local emergency services (112/911) immediately or press the SOS button on your dashboard. Do NOT delay seeking help."
        if any(w in msg for w in ['blood pressure', 'bp', 'hypertension']):
            return "Normal blood pressure is around 120/80 mmHg. High blood pressure (above 140/90) increases heart disease risk. Reduce salt intake, exercise regularly, and consult your doctor for medication if needed. (I am an AI - please consult your HeartGuard doctor for a clinical plan.)"
        if any(w in msg for w in ['cholesterol', 'ldl', 'hdl']):
            return "Healthy total cholesterol is below 200 mg/dL. HDL (good) should be above 40 mg/dL, LDL (bad) below 100 mg/dL. Eat more fiber, omega-3 fatty acids, and exercise regularly. (I am an AI - please consult your HeartGuard doctor for a clinical plan.)"
        if any(w in msg for w in ['diet', 'food', 'eat']):
            return "A heart-healthy diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats (olive oil, nuts, fish). Limit salt, sugar, red meat, and processed foods. The DASH and Mediterranean diets are clinically recommended. (I am an AI - please consult your HeartGuard doctor.)"
        if any(w in msg for w in ['exercise', 'walk', 'workout', 'physical']):
            return "Aim for at least 150 minutes of moderate aerobic activity per week (brisk walking, cycling, swimming). Even 30 minutes of daily walking significantly reduces heart disease risk. Start slow and increase gradually. (I am an AI - please consult your HeartGuard doctor.)"
        if any(w in msg for w in ['stress', 'anxiety', 'tension']):
            return "Chronic stress raises blood pressure and heart rate. Try deep breathing exercises, meditation, yoga, or short walks. Good sleep (7-8 hours) also helps manage stress. (I am an AI - please consult your HeartGuard doctor.)"
        if any(w in msg for w in ['smoking', 'tobacco', 'cigarette']):
            return "Smoking greatly increases heart disease risk. Quitting smoking is the single best thing you can do for your heart. Benefits start within 20 minutes of your last cigarette. Ask your doctor about nicotine replacement therapy. (I am an AI - please consult your HeartGuard doctor.)"
        if any(w in msg for w in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
            return "Hello! I'm HeartGuard AI, your 24/7 cardiac health assistant. Ask me about blood pressure, cholesterol, heart-healthy diets, exercise, or any heart health concern. (Note: I am an AI and cannot diagnose or prescribe - please consult your doctor for clinical plans.)"
        return "Thank you for your question! For general heart health: maintain a balanced diet, exercise 30 minutes daily, manage stress, avoid smoking, and monitor your blood pressure regularly. For specific medical advice, please consult your HeartGuard doctor. (I am an AI health assistant - not a substitute for professional medical care.)"
    
    try:
        model = _genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_chatbot_prompt())
        history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in conversation_history])
        full_prompt = f"{history_text}\nuser: {user_message}\nmodel:" if history_text else user_message
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Chat Error: {e}")
        return "I'm having trouble connecting to my knowledge base right now. Please try again later."
