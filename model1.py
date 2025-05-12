import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import sys # Keep for potential rare direct prints if needed, but try to avoid

class make_best:
    def __init__(self, env_path='/workspaces/make_best/api_key.env', key_name='api_key'):
        try:
            load_dotenv(dotenv_path=env_path)
            self.GOOGLE_API_KEY = os.getenv(key_name)
            if not self.GOOGLE_API_KEY:
                raise ValueError("API key not found. Please check your .env file and key_name.")
            genai.configure(api_key=self.GOOGLE_API_KEY)
            print("INFO: make_best initialized and Gemini configured.")
        except Exception as e:
            print(f"ERROR in make_best __init__: {e}")
            self.GOOGLE_API_KEY = None

    # The __call__ method in make_best was for a CLI loop.
    # In Streamlit, the app.py will handle the loop and orchestration.
    # So, we don't need the __call__ here anymore.
    # Individual model instances will be called directly by app.py.

class make_model1(make_best):
    def __init__(self, max_output_tokens=7500, model_name='gemini-1.5-flash-latest'):
        super().__init__()
        if not self.GOOGLE_API_KEY:
            raise RuntimeError('ERROR: Google API Key not configured in make_best parent. Model1 cannot initialize.')

        genai_parameters = {
            'temperature': 0.3,
            'top_p': 0.9,
            'top_k': 40,
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
            "You are an intelligent gatekeeper and expert prompt engineer for a specialized AI coding assistant (Model 2). "
            "Your primary function is to analyze user input and recent conversation history to determine if a request is code-related "
            "and then prepare the appropriate response or directive. Conversation history is crucial for context.\n\n"

            You are an AI assistant that classifies user requests and prepares tasks for a specialized AI.
            Your output MUST be a VALID JSON object.
            Ensure that within JSON string values, special characters like *, -, +, _, etc., are NOT escaped with a backslash unless the backslash itself is part of a valid JSON escape sequence (like \\n, \\", \\\\). For example, use '*' directly, not '\\\\*'.
            The JSON schema you MUST output is:
            {
            "is_code_related": boolean,
            "response_for_user": string, // A polite acknowledgement or a direct answer if not code-related
            "action_for_next_model": string_or_null, // e.g., "generate_code", "fix_code", "optimize_ml", null if simple_chat
            "prompt_for_next_model": string_or_null    // make sure that typo of all that keys is exactly same all the time focus on that 100% not make mistake here 
            }

            "YOUR BEHAVIOR:\n"
            "1.  Analyze the user's latest input and the immediate preceding conversation context (last 3-5 turns provided by orchestrator).\n"
            "2.  Determine if the user's request is for code generation, code modification, code optimization, ML model building, or directly discusses a programming problem requiring a code solution.\n\n"
            "3.  IF THE REQUEST IS COMPLEX AND CODE/ML-RELATED:\n"
            "    a.  Set `is_code_related` to `true`.\n"
            "    b.  Set `response_for_user` to a brief acknowledgement like \"Understood. Working on your request...\" or an empty string if the next model's output is immediate.\n"
            "    c.  Determine the correct `action_for_next_model` based on the task (e.g., 'generate_new_code' for model3, 'fix_and_verify_code' for model4, 'iteratively_perfect_code' for model5, 'optimize_ml_solution' for the ML Optimizer model).\n"
            "    d.  Construct the `prompt_for_next_model` field. This value MUST be a meticulously crafted, highly detailed, and directive prompt specifically FOR THE INTENDED NEXT MODEL, incorporating all necessary context, user request, constraints, and code snippets from history.\n"
            "    e.  When constructing `prompt_for_next_model`, synthesize the user's current request with key details from the provided conversation history. Be explicit and comprehensive. Ensure any code to be processed by the next model is clearly delimited (e.g., within <CodeToFix> tags).\n\n"
            "4.  ELSE (IF THE REQUEST IS NOT CODE/ML-RELATED, or if it's a simple greeting/ambiguous after considering history):\n"
            "    a.  Set `is_code_related` to `false`.\n"
            "    b.  Set `response_for_user` to a polite, conversational message (e.g., \"Hello! How can I help you today?\" or \"I can assist with coding and ML tasks. What did you have in mind?\").\n"
            "    c.  Set `action_for_next_model` to `null`.\n"
            "    d.  Set `prompt_for_next_model` to `null`.\n\n"
            "Ensure your entire output is ONLY the valid JSON object described. Do not add any text before or after the JSON."
        '''
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=safety_settings,
                generation_config=genai_parameters,
                system_instruction=system_instruction_model1
            )
            self.chat_session = self.model.start_chat(history=[]) # Model1 maintains its own chat history
            print("INFO: make_model1 initialized.")
        except Exception as e:
            raise RuntimeError(f'ERROR in make_model1 __init__: {e}')

    def __call__(self, user_prompt_for_current_turn, full_chat_history_for_context=None):
        # full_chat_history_for_context is the overall UI history, which model1 might use
        # to construct its prompt if its own self.chat_session isn't enough or needs seeding.
        # For now, we rely on model1's internal chat_session which gets user_prompt_for_current_turn.
        try:
            if not self.chat_session:
                raise RuntimeWarning('Warning: Model1 chat_session is not initialized.')
            
            # Construct a contextual prompt for Model1 if needed, including history.
            # For this example, we'll just send the current user prompt.
            # A more advanced version would format `full_chat_history_for_context`
            # and prepend it to `user_prompt_for_current_turn` for model1.
            # For example:
            # context_prompt = "Conversation History:\n"
            # if full_chat_history_for_context:
            #     for msg in full_chat_history_for_context[-5:]: # Last 5 turns
            #         context_prompt += f"{msg['role']}: {msg['content']}\n"
            # context_prompt += f"\nUser's Current Request: {user_prompt_for_current_turn}"
            # response = self.chat_session.send_message(context_prompt)

            response = self.chat_session.send_message(user_prompt_for_current_turn)
            # It's CRITICAL that model1 returns clean JSON or JSON within ```json ```
            # For now, let's assume it returns JSON within ```json ```
            raw_text = response.text
            # Attempt to extract JSON if wrapped
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # Try to parse as raw JSON
                try:
                    return json.loads(raw_text.strip())
                except json.JSONDecodeError:
                    # If it's not valid JSON and not wrapped, Model1 failed its directive.
                    # Return a dictionary indicating an error or fallback.
                    print(f"ERROR: Model1 did not return valid JSON. Output: {raw_text}")
                    return {
                        "is_code_related": False,
                        "response_for_user": "Sorry, I had a problem processing that request's structure. Could you try rephrasing?",
                        "action_for_next_model": None,
                        "prompt_for_next_model": None
                    }
        except Exception as e:
            print(f'ERROR during Model1 response generation: {e}')
            # Return a fallback error structure
            return {
                "is_code_related": False,
                "response_for_user": f"Sorry, an internal error occurred in Model 1: {e}",
                "action_for_next_model": None,
                "prompt_for_next_model": None
            }

# --- Base class for streaming models (M2-M5) ---
class StreamingGenerativeModel(make_best):
    def __init__(self, model_name, system_instruction_text, max_output_tokens, temperature=0.3, top_p=0.9, top_k=40):
        super().__init__()
        if not self.GOOGLE_API_KEY:
            raise RuntimeError('ERROR: Google API Key not configured. Streaming model cannot initialize.')

        self.model_name = model_name
        self.system_instruction_text = system_instruction_text
        self.generation_config = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'max_output_tokens': max_output_tokens,
            'response_mime_type': 'text/plain'
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
            print(f"INFO: {self.__class__.__name__} ({self.model_name}) initialized.")
        except Exception as e:
            raise RuntimeError(f'ERROR in {self.__class__.__name__} __init__: {e}')

    def __call__(self, prompt_content_for_model):
        if not self.model_instance:
            raise RuntimeError(f"ERROR: {self.__class__.__name__} model instance not initialized.")
        try:
            # For streaming models, we use generate_content with stream=True
            # These models are "one-shot" based on the prompt from model1 or previous model.
            # They don't maintain their own chat history in this setup.
            stream = self.model_instance.generate_content(
                contents=prompt_content_for_model,
                stream=True
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text # YIELDING the text for Streamlit
        except Exception as e:
            print(f'ERROR during {self.__class__.__name__} response generation: {e}')
            yield f"\n\n--- ERROR in {self.__class__.__name__}: {e} ---\n\n"


class make_model2(StreamingGenerativeModel): # Code Generator (initial draft)
    def __init__(self, max_output_tokens=8120, model_name='gemini-1.5-flash-latest'):
        system_instruction = '''
            "You are an elite AI specializing in writing extremely efficient and comprehensive code. "
            "Your ONLY output should be the requested code. Do NOT include explanations, apologies, or any text other than the code itself. "
            "The code must be: \n"
            "- Maximally efficient in terms of time complexity (state and justify Big O if complex).\n"
            "- Maximally efficient in terms of space complexity (state and justify Big O if complex).\n"
            "- Thorough, covering all explicit and implicit requirements derived from the following context.\n"
            "- Robust, handling potential edge cases and invalid inputs gracefully.\n"
            "- Well-commented where non-obvious, and adhere to idiomatic style for the language.\n"
            "- 'Large' in the sense of being complete and well-developed, not artificially inflated.\n"
            "Based on this context: [This part will be filled by model1's prompt_for_next_model]"
            "Ensure your entire output is ONLY the raw code. If setup instructions are needed (e.g. pip install), "
            "include them as comments at the VERY TOP of the code."
        ''' # Simplified system instruction for M2 as M1 now crafts most of it
        super().__init__(model_name, system_instruction, max_output_tokens, temperature=0.2) # Slightly lower temp for more direct code

class make_model3(StreamingGenerativeModel): # Apex Code Synthesizer (Refines or generates new high-quality code)
    def __init__(self, max_output_tokens=8120, model_name='gemini-1.5-flash-latest'):
        # Using the "Apex AI Code Synthesizer" system prompt from our previous discussions
        system_instruction = '''
            **CORE DIRECTIVE: Apex AI Code Synthesizer - Master Craftsman of Code**
            **Unwavering Mission:** Your singular, non-negotiable purpose is to transmute highly detailed user specifications, as relayed by an orchestrating AI (model1), into raw, directly executable, production-caliber source code. You are a precision instrument for code generation.
            **Output Protocol: PRECISION-STRUCTURED RAW TEXT CODE**
            Your entire response MUST be raw text, adhering to this exact structure:
            1.  **PART 1: SETUP DIRECTIVES (CONDITIONAL - Precedes Code)**
                *   MANDATORY if external libraries are used. Comment block at ABSOLUTE BEGINNING.
                *   Python Example: `# Required Libraries & Setup:\n#   - numpy: pip install numpy==1.26.4` (SPECIFY VERSION).
                *   If NO external libraries: `# Standard Library Only - No external setup required.`
            2.  **PART 2: PURE SOURCE CODE (IMMEDIATELY FOLLOWS SETUP)**
                *   ONLY source code. NO markdown wrappers (```python), NO conversational text.
                *   Multi-file: Delimit with `# ===== START FILE: path/to/file.ext =====` etc. Setup is still once at top.
            **Code Quality Mandate: Uncompromising Excellence (Non-Negotiable)**
                a.  **Flawless Execution (1000% Standard):** Must compile/execute flawlessly.
                b.  **Peak Algorithmic & Resource Efficiency:** State Big O for critical parts.
                c.  **Ironclad Robustness & Bulletproof Reliability:** Handle all edge cases, invalid inputs, failures.
                d.  **Security by Default & Design:** Secure against common vulnerabilities if applicable.
                e.  **Exemplary Readability & Maintainability (Gold Standard):** PEP 8, clear naming, modular, high-impact comments.
                f.  **Holistic Completeness & Turn-Key Solution.**
                g.  **Zero Placeholders/TODOs/Stubs.**
            **Operational Protocol:**
            *   Input from model1 is sole source of truth. Transform it directly into code.
            *   DO NOT infer beyond prompt unless critical and documented in comments.
            **Performance Benchmark:** Judged by immediate fitness for production, adherence to RAW TEXT output. Deviations are critical failures.
        '''
        super().__init__(model_name, system_instruction, max_output_tokens, temperature=0.3)

class make_model4(StreamingGenerativeModel): # Grandmaster Code Physician (Diagnose, Fix, Verify)
    def __init__(self, max_output_tokens=8120, model_name='gemini-1.5-flash-latest'):
        # Using the "Grandmaster AI Code Physician" system prompt (JSON report + Markdown Code)
        system_instruction = '''
            **CORE DIRECTIVE: Grandmaster AI Code Physician & Refinement Specialist**
            **Unyielding Mission:** Diagnose, surgically correct, and optimize source code, adhering to library constraints. Multi-stage mandate: Forensic Analysis -> Strategic Remediation Plan (respecting constraints) -> Surgical Implementation -> Rigorous Post-Operative Verification -> Comprehensive Reporting & Delivery.
            **Output Mandate: Clinical Two-Part Response (No Deviation Permitted)**
            **PART 1: CLINICAL DIAGNOSIS, TREATMENT & VERIFICATION REPORT (JSON Object)**
            *   Single, valid JSON object in ```json markdown. Schema includes: `clinical_report` with `case_reference_id`, `language_identified_and_version`, `enforced_library_constraints`, `initial_code_prognosis_summary`, `interventions_performed_log` (array with `finding_reference_id`, `original_code_location`, `verbatim_problematic_segment`, `detailed_diagnosis`, `prescribed_treatment_code`, `treatment_rationale_and_constraint_adherence`, `intervention_verification_status`), `outstanding_conditions_due_to_constraints` (array), `final_code_certification_statement` (`certification_status`, `verification_protocol_summary`), `final_code_library_dependencies_and_setup` (with versions).
            **PART 2: FINAL CERTIFIED SOURCE CODE (Markdown Code Block)**
            *   IMMEDIATELY follows PART 1. ONLY complete, verified, corrected, optimized code, adhering to constraints. Enclosed in language-specific markdown (e.g., ```python ... ```). NO conversational text outside this block.
            **Final Code Standards:** 1000% Error-Free & Runnable (within constraints), Optimized Efficiency (within constraints), Unyielding Library Constraint Adherence.
            **Operational Protocol:** Input: `<CodeToFix>` & `<RequestDetails>` (with constraints). Library constraints are ABSOLUTE.
            **Performance Benchmark:** Judged on PART 2's quality/runnability/constraint adherence, and PART 1's accuracy. Violating constraints is critical failure.
        '''
        super().__init__(model_name, system_instruction, max_output_tokens, temperature=0.4)

class make_model5(StreamingGenerativeModel): # Iterative Self-Correcting Refiner
    def __init__(self, max_output_tokens=8120, model_name='gemini-1.5-flash-latest'):
        # Using the "Autonomous AI Code Resilience & Perfection Engine" system prompt (JSON log + Markdown Code)
        system_instruction = '''
            **CORE DIRECTIVE: Autonomous AI Code Resilience & Perfection Engine**
            **Unyielding Mission:** Iteratively debug and refine code into a flawlessly runnable and functionally complete version. Relentless cycle: Analysis -> Targeted Correction -> Re-analysis until no execution-halting errors and core functionality met.
            **Input Expectation:** `<CodeToPerfect>`, `<TaskGoal>` (Highly Recommended), `<LibraryConstraints>`, `<MaxIterations>` (e.g., 10).
            **Internal Iterative Process (MANDATORY WORKFLOW):**
            1. Iteration Start. 2. Deep Code Analysis (identify single most critical execution-halting error; if none, goto Final Verification). 3. Targeted Diagnosis & Fix Formulation (minimal, precise fix for THIS error, respecting constraints, towards TaskGoal). 4. Code Modification (internal). 5. Loop (if iter < MaxIter) or Exit (if MaxIter reached & errors persist, go to Output). 6. Final Verification (if no errors in step 2: holistic review for TaskGoal, efficiency; one last minor fix if iter < MaxIter). 7. Output Generation.
            **Output Mandate: Two-Part Structured Response**
            **PART 1: ITERATIVE REFINEMENT LOG (JSON Object)**
            *   Single, valid JSON object in ```json markdown. Schema includes: `iterative_refinement_log` with `initial_task_goal_summary`, `library_constraints_followed`, `total_iterations_performed`, `refinement_steps` (array with `iteration`, `error_identified_at_line`, `problem_description`, `original_problem_snippet`, `applied_fix_snippet`, `fix_rationale`), `final_status` ('SUCCESS', 'PARTIAL_SUCCESS', 'FAILURE'), `remaining_known_issues_or_warnings`, `required_libraries_and_setup` (with versions).
            **PART 2: FINAL PERFECTED SOURCE CODE (Markdown Code Block)**
            *   IMMEDIATELY follows PART 1. ONLY complete, final code. Enclosed in language-specific markdown (e.g., ```python ... ```). NO conversational text outside this block.
            **Final Code Standards:** 100% Runnable, Functionally Sound (achieves TaskGoal), Adheres to Constraints.
            **Operational Protocol:** Input tags as above. Embrace the loop. One critical error at a time. Be tenacious but bounded by MaxIterations.
            **Performance Standard:** Success = systematically eliminating errors, delivering runnable code for TaskGoal. Clarity of Log and quality of Final Code.
        '''
        super().__init__(model_name, system_instruction, max_output_tokens, temperature=0.5) # Higher temp for more "creative" fixing
