import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re # For parsing model1's output if necessary
from pathlib import Path

# --- Base Class for API Key and Basic Config ---
class AIModelBase:
    def __init__(self, env_path=None, key_name='api_key'):
        self.GOOGLE_API_KEY = None
        try:
            if env_path is None:
                # Construct path to api_key.env assuming it's in the same directory as this script
                script_dir = Path(__file__).resolve().parent
                env_path_to_load = script_dir / 'api_key.env'
            else:
                env_path_to_load = Path(env_path)

            print(f"INFO: Attempting to load .env file from: {env_path_to_load}")
            if env_path_to_load.is_file():
                load_dotenv(dotenv_path=env_path_to_load)
                print(f"INFO: Loaded .env file from: {env_path_to_load}")
            else:
                print(f"WARNING: .env file not found at {env_path_to_load}. "
                      "Will rely on environment variables if set.")

            self.GOOGLE_API_KEY = os.getenv(key_name)

            if not self.GOOGLE_API_KEY:
                error_message = (
                    f"API key '{key_name}' not found. "
                    f"Checked path: '{env_path_to_load}'. "
                    "Ensure the file exists and contains the key, or that the environment variable is correctly set."
                )
                print(f"ERROR: {error_message}")
                raise ValueError(error_message)

            genai.configure(api_key=self.GOOGLE_API_KEY)
            print(f"INFO: AIModelBase initialized and Gemini configured successfully using key '{key_name}'.")

        except ValueError as ve:
            print(f"CONFIG ERROR in AIModelBase __init__: {ve}")
            # Re-raise as RuntimeError to be caught by Streamlit or higher-level handlers
            raise RuntimeError(f"API Key Configuration Error: {ve}") from ve
        except Exception as e:
            print(f"UNEXPECTED ERROR in AIModelBase __init__: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Unexpected Initialization Error: {e}") from e

# --- Model 1: Orchestrator ---
class make_model1(AIModelBase):
    def __init__(self, model_name='gemini-1.5-flash-latest', max_output_tokens=2048): # Reduced max tokens for M1
        super().__init__() # Initializes API key
        if not self.GOOGLE_API_KEY: # Check again after super()
            raise RuntimeError('ERROR: Google API Key not configured. Model1 cannot initialize.')

        self.model_name = model_name
        genai_parameters = {
            'temperature': 0.2, # Lower for more deterministic JSON output
            'top_p': 0.9,
            'top_k': 30,
            'max_output_tokens': max_output_tokens,
            'response_mime_type': 'text/plain' # Expecting JSON string, but model returns text
        }
        safety_settings = [
            {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
            {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_MEDIUM_AND_ABOVE'}
        ]
        # Corrected escape sequence for '\*' -> '\\*'
        system_instruction_model1 = '''
            You are an AI Orchestrator. Your primary function is to analyze user input and recent conversation history (if provided) to determine the user's intent and prepare a structured JSON directive for subsequent specialized AI models.

            Your output MUST BE a single, VALID JSON object. NOTHING ELSE.
            Ensure that within JSON string values, special characters like *, -, +, _, etc., are NOT escaped with a backslash unless it's part of a valid JSON escape sequence (e.g., \\n, \\", \\\\). For example, use '*' directly, not '\\*'.

            The JSON schema you MUST output is:
            {
              "is_code_related": boolean, // True if the request is about coding, ML, algorithms, etc.
              "user_facing_acknowledgement": string, // A brief, polite acknowledgement for the user (e.g., "Okay, looking into that..." or "Hello!"). Can be an empty string if the next model's output is the primary response.
              "action_for_next_model": string_or_null, // "generate_new_code_m3", "fix_and_verify_code_m4", "iteratively_perfect_code_m5", "optimize_ml_solution_m_ml", or null for simple chat.
              "prompt_for_next_model": string_or_null, // The detailed prompt for the specialized model if action is not null.
              "library_constraints_for_next_model": string_or_null // e.g., "Python standard library only", "NumPy and Pandas", null if not specified by user.
            }

            BEHAVIOR:
            1.  Analyze the user's LATEST input. Consider provided conversation HISTORY for context.
            2.  Determine intent: Simple chat? Code generation? Code fixing? ML task?
            3.  Populate the JSON fields accurately based on your analysis.

            EXAMPLES of your thought process for `action_for_next_model` and `prompt_for_next_model`:

            IF USER ASKS TO "generate a python script for X":
                "action_for_next_model": "generate_new_code_m3",
                "prompt_for_next_model": "<RequestDetails>User wants a Python script for X. Ensure comprehensive, runnable code with setup instructions.</RequestDetails><LibraryConstraints>[Extract or infer constraints, e.g., 'Python standard library only' if none mentioned]</LibraryConstraints>"

            IF USER PROVIDES CODE AND SAYS "fix this error" or "improve this code":
                "action_for_next_model": "fix_and_verify_code_m4", // Or "iteratively_perfect_code_m5" for more complex cases
                "prompt_for_next_model": "<CodeToFix language='[infer_language]'>\n[PASTE_USER_CODE_HERE]\n</CodeToFix>\n<RequestDetails>User wants this code fixed/improved: [USER_SPECIFIC_REQUEST_ABOUT_FIXING]. Adhere to library constraints: [CONSTRAINTS].</RequestDetails><LibraryConstraints>[Extract constraints, e.g., 'NumPy only']</LibraryConstraints>"

            IF USER ASKS "make a classification model for highly imbalanced data":
                "action_for_next_model": "optimize_ml_solution_m_ml",
                "prompt_for_next_model": "<MLTaskDescription>User wants a classification model for highly imbalanced data (e.g., 90:10). Focus on F1/AUC, robust preprocessing, and overfitting prevention. Language: Python. Libraries: scikit-learn, imblearn unless specified otherwise.</MLTaskDescription><OriginalCodeContext>[Include if user provided starting code]</OriginalCodeContext><LibraryConstraints>[User-specified library constraints]</LibraryConstraints>"

            IF USER SAYS "hello" or asks a general question:
                "is_code_related": false,
                "user_facing_acknowledgement": "Hello! How can I help you today?",
                "action_for_next_model": null,
                "prompt_for_next_model": null

            CRITICAL: Extract any library constraints (e.g., "only use numpy") from the user's request or history and place them in `library_constraints_for_next_model`.
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
            # Model1 uses a chat session to maintain context for its own decision-making process
            # if it needs to ask clarifying questions or have a multi-turn interaction
            # before deciding on the next action.
            self.chat_session = self.model_instance.start_chat(history=[])
            print("INFO: make_model1 initialized.")
        except Exception as e:
            raise RuntimeError(f'ERROR in make_model1 __init__: {e}')

    def __call__(self, user_prompt_for_current_turn, ui_chat_history_for_context=None):
        """
        Processes the user's current prompt, considering overall UI chat history for context.
        Returns a Python dictionary parsed from the LLM's JSON output.
        """
        try:
            if not self.chat_session:
                raise RuntimeError('Model1 chat_session is not initialized.')

            # Construct a more contextual prompt for Model1, including recent UI history
            contextual_prompt_for_model1 = f"Current user request: \"{user_prompt_for_current_turn}\"\n\n"
            if ui_chat_history_for_context and len(ui_chat_history_for_context) > 1: # Exclude initial assistant message
                contextual_prompt_for_model1 += "Recent conversation history (last 3 turns):\n"
                # Get last 3 exchanges (user + assistant = 1 exchange, so 6 messages max)
                for msg_data in ui_chat_history_for_context[-7:-1]: # Last 3 user/assistant pairs + current user prompt
                    # Extract text content from potentially structured messages
                    text_content = ""
                    if "content_parts" in msg_data:
                        for part in msg_data["content_parts"]:
                            if part["type"] == "text":
                                text_content += part["data"] + " "
                    elif "content" in msg_data: # Fallback for simple string content
                        text_content = msg_data["content"]

                    if text_content.strip(): # Only add if there's actual text
                        contextual_prompt_for_model1 += f"{msg_data['role']}: {text_content.strip()}\n"
            
            contextual_prompt_for_model1 += f"\nBased on the current request and history, generate your JSON directive."
            
            # print(f"DEBUG: Prompt to Model1:\n{contextual_prompt_for_model1}") # For debugging

            # Send to Model1's internal chat session
            response = self.chat_session.send_message(contextual_prompt_for_model1)
            raw_text = response.text.strip()

            # Attempt to extract JSON if wrapped in markdown
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(1)
            else:
                json_string = raw_text # Assume raw JSON if no markdown wrapper

            try:
                parsed_json = json.loads(json_string)
                # Validate essential keys
                if not all(k in parsed_json for k in ["is_code_related", "user_facing_acknowledgement", "action_for_next_model", "prompt_for_next_model"]):
                    print(f"WARNING: Model1 JSON missing essential keys. Output: {parsed_json}")
                    # Fallback if keys are missing
                    raise json.JSONDecodeError("Essential keys missing from Model1 JSON", json_string, 0)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"ERROR: Model1 did not return valid JSON. Error: {e}. Raw output: '{raw_text}'")
                return {
                    "is_code_related": False,
                    "user_facing_acknowledgement": f"Sorry, I had a problem understanding the request structure (M1_JSON_ERR). Raw: {raw_text[:100]}...",
                    "action_for_next_model": None,
                    "prompt_for_next_model": None,
                    "library_constraints_for_next_model": None
                }
        except Exception as e:
            print(f'ERROR during Model1 response generation: {e}')
            import traceback
            traceback.print_exc()
            return {
                "is_code_related": False,
                "user_facing_acknowledgement": f"Sorry, an internal error occurred in Model 1: {e}",
                "action_for_next_model": None,
                "prompt_for_next_model": None,
                "library_constraints_for_next_model": None
            }


# --- Base Class for Specialized Streaming Models (M2-M5, ML_Optimizer) ---
class SpecializedStreamingModel(AIModelBase):
    def __init__(self, model_name, system_instruction_text, max_output_tokens, temperature=0.3, top_p=0.9, top_k=40):
        super().__init__() # Initializes API key
        if not self.GOOGLE_API_KEY:
            raise RuntimeError('ERROR: Google API Key not configured. Specialized model cannot initialize.')

        self.model_name = model_name
        self.system_instruction_text = system_instruction_text
        self.generation_config = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'max_output_tokens': max_output_tokens,
            'response_mime_type': 'text/plain'
        }
        self.safety_settings = [ # Standard safety settings
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
        """
        Generates content in a streaming fashion.
        Yields text chunks.
        """
        if not self.model_instance:
            raise RuntimeError(f"ERROR: {self.__class__.__name__} model instance not initialized.")
        try:
            # These models are generally "one-shot" for their specific task based on the detailed prompt.
            stream = self.model_instance.generate_content(
                contents=prompt_content_for_model, # This is the detailed prompt from Model1 or previous stage
                stream=True
            )
            for chunk_idx, chunk in enumerate(stream):
                if chunk.text:
                    yield chunk.text
                # Handle potential stream interruption or errors from Gemini
                if not chunk.candidates: # Check if there are candidates
                     print(f"WARNING: Chunk {chunk_idx} from {self.__class__.__name__} has no candidates. Finish reason: {chunk.prompt_feedback if chunk.prompt_feedback else 'N/A'}")
                     if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                         yield f"\n\n---STREAM BLOCKED by Safety Filter in {self.__class__.__name__}: {chunk.prompt_feedback.block_reason_message}---\n"
                     # Consider breaking or yielding an error message
                     break # Stop processing this stream if it's blocked or empty
                
                # You might want to check chunk.finish_reason as well inside the loop if it's relevant
                # For example, if stream.candidates[0].finish_reason == "MAX_TOKENS":
                #    yield "\n\n---MAX_TOKENS_REACHED---\n"

        except Exception as e:
            print(f'ERROR during {self.__class__.__name__} streaming response: {e}')
            import traceback
            traceback.print_exc()
            yield f"\n\n--- ERROR in {self.__class__.__name__}: {e} ---\n\n"

# --- Specialized Model Implementations ---

class make_model2(SpecializedStreamingModel): # Simple Code Generator (if M1 routes to it directly)
    def __init__(self, max_output_tokens=8120, model_name='gemini-1.5-flash-latest'):
        # This system prompt is for a basic code generator. Model1 would have provided detailed context.
        system_instruction = '''
            You are an AI Code Generator. Your SOLE output is raw, executable code based on the user's request.
            If libraries are needed, include installation commands as comments at the VERY TOP (e.g., `# pip install library`).
            Do NOT include any conversational text, explanations, or markdown wrappers like ```python.
            Focus on correctness, efficiency, and readability.
        '''
        super().__init__(model_name, system_instruction, max_output_tokens, temperature=0.2)

class make_model3(SpecializedStreamingModel): # Apex Code Synthesizer
    def __init__(self, max_output_tokens=8120, model_name='gemini-1.5-flash-latest'):
        # PASTE YOUR FULL "Apex AI Code Synthesizer" SYSTEM INSTRUCTION HERE
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
        ''' # Ensure this is the full, correct prompt
        super().__init__(model_name, system_instruction, max_output_tokens, temperature=0.3)

class make_model4(SpecializedStreamingModel): # Grandmaster Code Physician
    def __init__(self, max_output_tokens=8120, model_name='gemini-1.5-flash-latest'):
        # PASTE YOUR FULL "Grandmaster AI Code Physician" SYSTEM INSTRUCTION HERE
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
        ''' # Ensure this is the full, correct prompt
        super().__init__(model_name, system_instruction, max_output_tokens, temperature=0.4)

class make_model5(SpecializedStreamingModel): # Iterative Self-Correcting Refiner
    def __init__(self, max_output_tokens=8120, model_name='gemini-1.5-flash-latest'):
        # PASTE YOUR FULL "Autonomous AI Code Resilience & Perfection Engine" SYSTEM INSTRUCTION HERE
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
        ''' # Ensure this is the full, correct prompt
        super().__init__(model_name, system_instruction, max_output_tokens, temperature=0.5)


class make_model_ml_optimizer(SpecializedStreamingModel): # ML Performance Optimizer
    def __init__(self, max_output_tokens=8192, model_name='gemini-1.5-flash-latest'): # Or gemini-1.5-pro for this
        # PASTE YOUR FULL "AI Peak Performance ML Engineering Specialist" SYSTEM INSTRUCTION HERE
        system_instruction = '''
            **CORE DIRECTIVE: AI Peak Performance ML Engineering Specialist**
            **Mission Critical Objective:** Transform a user's ML problem description—and any provided initial code—into a fully operational, robust, and **maximally performant** ML solution. Final code MUST be 100% runnable, achieve highest possible relevant metrics, be resilient against overfitting, and represent gold standard in ML engineering.
            **Input Expectation:** `<MLTaskDescription>` (detailed ML problem, dataset characteristics, performance metrics, algo preferences, constraints), `<OriginalCodeContext>` (Optional).
            **Output Mandate: ULTIMATE ML CODE SOLUTION & STRATEGY BRIEF**
            **PART 1: STRATEGIC OVERVIEW & ASSURANCE (Concise JSON Object)**
            *   Single, valid JSON object in ```json markdown. Schema: `ml_optimization_strategy_brief` with `task_understanding_and_objective`, `chosen_model_family_and_justification`, `key_optimization_levers` (array: hyperparam tuning, feature eng, imbalance handling, ensembles, cross-val), `overfitting_prevention_guarantee`, `expected_performance_tier`, `library_constraints_adherence`.
            **PART 2: REQUIRED LIBRARIES & SETUP (Comment Block in Final Code)**
            *   Comment block at VERY BEGINNING of PART 3. List all non-standard external libraries and install commands with EXACT VERSIONS (e.g., `# REQUIREMENTS:\n# pip install scikit-learn==1.3.2 pandas==2.0.3`). If only standard, state: `# No external non-standard libraries required.`
            **PART 3: THE ULTIMATE, 1000% WORKING, HIGH-PERFORMANCE ML CODE (Markdown Code Block)**
            *   IMMEDIATELY follows PART 2. ONLY complete, runnable, highly optimized, robust ML source code in language-specific markdown (e.g., ```python ... ```). NO conversational text outside block.
            *   **Code Quality & Performance Mandate:** 1000% Functionality & Zero Errors. Peak Predictive Performance (optimized for target metrics). Ironclad Overfitting Prevention. MANDATORY Reproducibility (fixed random seeds). Production-Grade Engineering (modular, PEP8, efficient data handling, clear logging, model saving/loading boilerplate). Adherence to Constraints. Complete Solution.
            **Operational Protocol:** Deeply analyze task & constraints -> Strategize for peak performance & robustness -> Design/Refine data pipeline -> Select/Design optimal model architecture -> Implement rigorous training & validation -> Ensure anti-overfitting -> Engineer for production -> Output structured response. Heavy refactor/rewrite of original code is authorized if it hinders peak performance (justify in PART 1).
            **Performance Standard:** Judged on PART 3's *demonstrable potential* for SOTA results, flawless execution, robustness, and strict adherence to output format. Deliver excellence.
        ''' # Ensure this is the full, correct prompt
        super().__init__(model_name, system_instruction, max_output_tokens, temperature=0.4) # Temp might vary for ML tasks
