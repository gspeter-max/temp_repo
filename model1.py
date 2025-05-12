import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
from pathlib import Path

# --- Base Class for API Key and Basic Config ---
class AIModelBase:
    def __init__(self, env_path=None, key_name='api_key'):
        self.GOOGLE_API_KEY = None
        try:
            if env_path is None:
                script_dir = Path(__file__).resolve().parent
                env_path_to_load = script_dir / 'api_key.env'
            else:
                env_path_to_load = Path(env_path)

            # print(f"INFO (AIModelBase): Attempting to load .env file from: {env_path_to_load}")
            if env_path_to_load.is_file():
                load_dotenv(dotenv_path=env_path_to_load)
                # print(f"INFO (AIModelBase): Loaded .env file from: {env_path_to_load}")
            # else:
                # print(f"WARNING (AIModelBase): .env file not found at {env_path_to_load}. Will rely on environment variables.")

            self.GOOGLE_API_KEY = os.getenv(key_name)

            if not self.GOOGLE_API_KEY:
                error_message = (
                    f"API key '{key_name}' not found. "
                    f"Checked path: '{env_path_to_load}'. "
                    "Ensure the file exists, contains the key, or the environment variable is correctly set."
                )
                print(f"ERROR (AIModelBase): {error_message}")
                raise ValueError(error_message)

            genai.configure(api_key=self.GOOGLE_API_KEY)
            # print(f"INFO (AIModelBase): Initialized and Gemini configured successfully.")

        except ValueError as ve:
            print(f"CONFIG ERROR in AIModelBase __init__: {ve}")
            raise RuntimeError(f"API Key Configuration Error: {ve}") from ve
        except Exception as e:
            print(f"UNEXPECTED ERROR in AIModelBase __init__: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Unexpected Initialization Error: {e}") from e

# --- Model 1: Orchestrator ---
class make_model1(AIModelBase):
    def __init__(self, model_name='gemini-1.5-flash-latest', max_output_tokens=2048):
        super().__init__()
        if not self.GOOGLE_API_KEY:
            raise RuntimeError('CRITICAL ERROR: Google API Key not configured from AIModelBase. Model1 cannot initialize.')

        self.model_name = model_name
        genai_parameters = {
            'temperature': 0.2,
            'top_p': 0.9,
            'top_k': 30,
            'max_output_tokens': max_output_tokens,
            'response_mime_type': 'text/plain'
        }
        safety_settings = [
            {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
            {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'}
        ]
        system_instruction_model1 = '''
            You are an AI Orchestrator. Your primary function is to analyze user input and recent conversation history (if provided) to determine the user's intent and prepare a structured JSON directive for subsequent specialized AI models.

            Your output MUST BE a single, VALID JSON object. NOTHING ELSE.
            Ensure that within JSON string values, special characters like *, -, +, _, etc., are NOT escaped with a backslash unless it's part of a valid JSON escape sequence (e.g., \\n, \\", \\\\). For example, use '*' directly, not '\\\\*'.

            The JSON schema you MUST output is:
            {
              "is_code_related": boolean, // True if the request is about coding, ML, algorithms, etc.
              "user_facing_acknowledgement": string, // A brief, polite acknowledgement for the user (e.g., "Okay, looking into that..." or "Hello!"). Can be an empty string if the next model's output is the primary response.
              "action_for_next_model": string_or_null, // "generate_new_code_m3", "fix_and_verify_code_m4", "iteratively_perfect_code_m5", "optimize_ml_solution_m_ml", or null for simple chat.
              "prompt_for_next_model": string_or_null, // The detailed prompt for the specialized model if action is not null.
              "library_constraints_for_next_model": string_or_null // e.g., "Python standard library only", "NumPy and Pandas", null if not specified by user.
            }

            BEHAVIOR:
            1. Analyze the user's LATEST input. Consider provided conversation HISTORY for context.
            2. Determine intent: Simple chat? Code generation? Code fixing? ML task?
            3. Populate the JSON fields accurately based on your analysis.

            EXAMPLES for `action_for_next_model` and `prompt_for_next_model`:
            IF USER ASKS TO "generate a python script for X":
                "action_for_next_model": "generate_new_code_m3",
                "prompt_for_next_model": "<RequestDetails>User wants a Python script for X. Ensure comprehensive, runnable code with setup instructions.</RequestDetails><LibraryConstraints>[Extract or infer constraints, e.g., 'Python standard library only' if none mentioned]</LibraryConstraints>"
            IF USER PROVIDES CODE AND SAYS "fix this error" or "improve this code":
                "action_for_next_model": "fix_and_verify_code_m4",
                "prompt_for_next_model": "<CodeToFix language='[infer_language]'>\\n[PASTE_USER_CODE_HERE]\\n</CodeToFix>\\n<RequestDetails>User wants this code fixed/improved: [USER_SPECIFIC_REQUEST_ABOUT_FIXING]. Adhere to library constraints: [CONSTRAINTS].</RequestDetails><LibraryConstraints>[Extract constraints, e.g., 'NumPy only']</LibraryConstraints>"
            IF USER ASKS "make a classification model for highly imbalanced data":
                "action_for_next_model": "optimize_ml_solution_m_ml",
                "prompt_for_next_model": "<MLTaskDescription>User wants a classification model for highly imbalanced data (e.g., 90:10). Focus on F1/AUC, robust preprocessing, and overfitting prevention. Language: Python. Libraries: scikit-learn, imblearn unless specified otherwise.</MLTaskDescription><OriginalCodeContext>[Include if user provided starting code]</OriginalCodeContext><LibraryConstraints>[User-specified library constraints]</LibraryConstraints>"
            IF USER SAYS "hello":
                "is_code_related": false, "user_facing_acknowledgement": "Hello! How can I help you today?", "action_for_next_model": null, "prompt_for_next_model": null, "library_constraints_for_next_model": null

            CRITICAL: Extract any library constraints and place in `library_constraints_for_next_model`.
            If `is_code_related` is true, `action_for_next_model` and `prompt_for_next_model` MUST NOT be null or empty.
            Your entire output is ONLY the JSON.
        '''
        try:
            self.model_instance = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=safety_settings,
                generation_config=genai_parameters,
                system_instruction=system_instruction_model1
            )
            self.chat_session = self.model_instance.start_chat(history=[])
            print("INFO (make_model1): Initialized.")
        except Exception as e:
            raise RuntimeError(f'ERROR in make_model1 __init__: {e}')

    def __call__(self, user_prompt_for_current_turn, ui_chat_history_for_context=None):
        try:
            if not self.chat_session:
                raise RuntimeError('Model1 chat_session is not initialized.')
            
            contextual_prompt_for_model1 = f"Current user request: \"{user_prompt_for_current_turn}\"\n\n"
            if ui_chat_history_for_context and len(ui_chat_history_for_context) > 1:
                contextual_prompt_for_model1 += "Recent conversation history (last 3 user/assistant exchanges):\n"
                history_to_send = []
                # Iterate backwards to get most recent, limit to 6 messages (3 exchanges)
                count = 0
                for msg_data in reversed(ui_chat_history_for_context[:-1]): # Exclude current user prompt
                    if count >= 6: break
                    text_content = ""
                    if "content_parts" in msg_data:
                        for part in msg_data["content_parts"]:
                            if part["type"] == "text": text_content += part["data"] + " "
                    elif "content" in msg_data: text_content = msg_data["content"]
                    if text_content.strip():
                        history_to_send.append(f"{msg_data['role']}: {text_content.strip()}")
                        count += 1
                contextual_prompt_for_model1 += "\n".join(reversed(history_to_send)) # Add in chronological order
            contextual_prompt_for_model1 += f"\n\nBased on the current request and this history, generate your JSON directive."
            
            response = self.chat_session.send_message(contextual_prompt_for_model1)
            raw_text = response.text.strip()

            json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if json_match: json_string = json_match.group(1)
            else: json_string = raw_text
            
            try:
                parsed_json = json.loads(json_string)
                if not all(k in parsed_json for k in ["is_code_related", "user_facing_acknowledgement", "action_for_next_model", "prompt_for_next_model", "library_constraints_for_next_model"]):
                    raise json.JSONDecodeError("Essential keys missing from Model1 JSON", json_string, 0)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"ERROR (make_model1): Did not return valid JSON. Error: {e}. Raw output: '{raw_text}'")
                return {
                    "is_code_related": False, "user_facing_acknowledgement": f"Sorry, I had a problem structuring my thoughts (M1_JSON_ERR).",
                    "action_for_next_model": None, "prompt_for_next_model": None, "library_constraints_for_next_model": None
                }
        except Exception as e:
            print(f'ERROR (make_model1) during response generation: {e}')
            import traceback; traceback.print_exc()
            return {
                "is_code_related": False, "user_facing_acknowledgement": f"Sorry, an internal error occurred in Model 1: {e}",
                "action_for_next_model": None, "prompt_for_next_model": None, "library_constraints_for_next_model": None
            }

# --- Base Class for Specialized Streaming Models ---
class SpecializedStreamingModel(AIModelBase):
    def __init__(self, class_name_for_log, model_name_suffix, system_instruction_text, max_output_tokens, temperature=0.3, top_p=0.9, top_k=40):
        super().__init__()
        if not self.GOOGLE_API_KEY:
            raise RuntimeError(f'CRITICAL ERROR: Google API Key not configured. {class_name_for_log} cannot initialize.')
        
        self.model_name = f'gemini-1.5-flash-{model_name_suffix}'
        self.system_instruction_text = system_instruction_text
        self.generation_config = {
            'temperature': temperature, 'top_p': top_p, 'top_k': top_k,
            'max_output_tokens': max_output_tokens, 'response_mime_type': 'text/plain'
        }
        self.safety_settings = [
            {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
            {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'}
        ]
        try:
            self.model_instance = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings,
                generation_config=self.generation_config,
                system_instruction=self.system_instruction_text
            )
            print(f"INFO ({class_name_for_log}): Initialized with model {self.model_name}.")
        except Exception as e:
            raise RuntimeError(f'ERROR in {class_name_for_log} __init__ for model {self.model_name}: {e}')

    def __call__(self, prompt_content_for_model):
        if not self.model_instance:
            raise RuntimeError(f"ERROR: {self.__class__.__name__} model instance not initialized.")
        try:
            stream = self.model_instance.generate_content(contents=prompt_content_for_model, stream=True)
            for chunk_idx, chunk in enumerate(stream):
                if chunk.text: yield chunk.text
                # It's important to check candidates, as an empty text might still have a finish reason
                if not chunk.candidates and chunk.prompt_feedback: # Check prompt_feedback if candidates are empty
                     if chunk.prompt_feedback.block_reason:
                         reason_message = chunk.prompt_feedback.block_reason_message or "Content blocked by safety filter"
                         print(f"WARNING: Stream from {self.__class__.__name__} blocked. Reason: {reason_message}")
                         yield f"\n\n---STREAM BLOCKED by Safety Filter in {self.__class__.__name__}: {reason_message}---\n"
                         break # Stop processing this stream if it's blocked
                elif chunk.candidates and chunk.candidates[0].finish_reason:
                    if chunk.candidates[0].finish_reason == genai.types.FinishReason.MAX_TOKENS:
                        yield "\n\n---MAX_TOKENS_REACHED---\n"
                        break
                    elif chunk.candidates[0].finish_reason not in [genai.types.FinishReason.FINISH_REASON_UNSPECIFIED, genai.types.FinishReason.STOP]:
                        # Other finish reasons like SAFETY, RECITATION, OTHER
                        yield f"\n\n---STREAM_ENDED_UNEXPECTEDLY ({chunk.candidates[0].finish_reason.name})---\n"
                        break


        except Exception as e:
            print(f'ERROR during {self.__class__.__name__} streaming response: {e}')
            import traceback; traceback.print_exc()
            yield f"\n\n--- ERROR in {self.__class__.__name__} while streaming: {e} ---\n\n"

# --- Specialized Model Implementations ---
class make_model2(SpecializedStreamingModel):
    def __init__(self, max_output_tokens=8120, model_name_suffix='latest'):
        system_instruction = """
You are an AI Code Generator. Your SOLE output is raw, executable code based on the user's request.
If libraries are needed, include installation commands as comments at the VERY TOP (e.g., `# pip install library`).
Do NOT include any conversational text, explanations, or markdown wrappers like ```python.
Focus on correctness, efficiency, and readability. The detailed requirements and context will be provided in the user prompt.
"""
        super().__init__("make_model2", model_name_suffix, system_instruction, max_output_tokens, temperature=0.2)

class make_model3(SpecializedStreamingModel): # Apex Code Synthesizer
    def __init__(self, max_output_tokens=8120, model_name_suffix='latest'):
        system_instruction = """[YOUR FULL 'Apex AI Code Synthesizer - Master Craftsman of Code' SYSTEM INSTRUCTION HERE - Outputs raw text: setup comments (with versions) then pure code, handles multi-file with specific delimiters]"""
        super().__init__("make_model3", model_name_suffix, system_instruction, max_output_tokens, temperature=0.3)

class make_model4(SpecializedStreamingModel): # Grandmaster Code Physician
    def __init__(self, max_output_tokens=8120, model_name_suffix='latest'):
        system_instruction = """[YOUR FULL 'Grandmaster AI Code Physician & Refinement Specialist' SYSTEM INSTRUCTION HERE - Outputs JSON report in ```json then code in ```language, respects library constraints meticulously]"""
        super().__init__("make_model4", model_name_suffix, system_instruction, max_output_tokens, temperature=0.4)

class make_model5(SpecializedStreamingModel): # Iterative Self-Correcting Refiner
    def __init__(self, max_output_tokens=8120, model_name_suffix='latest'):
        system_instruction = """[YOUR FULL 'Autonomous AI Code Resilience & Perfection Engine' SYSTEM INSTRUCTION HERE - Outputs JSON log in ```json then code in ```language, performs internal iteration]"""
        super().__init__("make_model5", model_name_suffix, system_instruction, max_output_tokens, temperature=0.5)

class make_model_ml_optimizer(SpecializedStreamingModel): # ML Performance Optimizer
    def __init__(self, max_output_tokens=8192, model_name_suffix='latest'):
        system_instruction = """[YOUR FULL 'AI Peak Performance ML Engineering Specialist' SYSTEM INSTRUCTION HERE - Outputs JSON strategy in ```json then ML code in ```language, focuses on SOTA performance and anti-overfitting]"""
        super().__init__("make_model_ml_optimizer", model_name_suffix, system_instruction, max_output_tokens, temperature=0.4)
