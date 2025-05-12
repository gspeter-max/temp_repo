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

            if env_path_to_load.is_file():
                load_dotenv(dotenv_path=env_path_to_load)

            self.GOOGLE_API_KEY = os.getenv(key_name)

            if not self.GOOGLE_API_KEY:
                error_message = (
                    f"API key '{key_name}' not found. Checked path: '{env_path_to_load}'. "
                    "Ensure the file exists, contains the key, or the environment variable is correctly set."
                )
                print(f"ERROR (AIModelBase): {error_message}")
                raise ValueError(error_message)

            genai.configure(api_key=self.GOOGLE_API_KEY)

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
            'temperature': 0.2, 'top_p': 0.9, 'top_k': 30,
            'max_output_tokens': max_output_tokens, 'response_mime_type': 'text/plain'
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
            2. Determine intent: Simple chat? Code generation? Code fixing (including ImportErrors)? ML task?
            3. Populate JSON fields accurately.

            SPECIFIC INSTRUCTIONS FOR `ImportError` SCENARIOS:
            IF the user provides code AND mentions an `ImportError` (e.g., "cannot import name X from Y"), or if you infer an ImportError from the context:
                Set "action_for_next_model": "fix_and_verify_code_m4".
                For "prompt_for_next_model", construct it to guide Model 4 thoroughly. Example:
                "<CodeToFix language='[lang]'>\\n[USER_CODE_WITH_IMPORT_ERROR]\\n</CodeToFix>\\n<ErrorContext>User reports: ImportError: cannot import name 'AdamW' from 'transformers'.</ErrorContext>\\n<RequestDetails>User needs to fix this ImportError. Your diagnosis MUST: 1. Confirm if the base library (e.g., 'transformers') is installed and provide install command if not. 2. CRITICALLY, investigate if 'AdamW' has been moved/deprecated in recent versions of 'transformers'. Provide the CORRECT, MODERN import path (e.g., from 'transformers.optimization' or 'torch.optim.AdamW' if more appropriate). 3. Provide the fully corrected code snippet. Ensure your JSON report details this version-aware diagnosis.</RequestDetails><LibraryConstraints>[Any existing constraints, e.g., 'Use transformers and torch']</LibraryConstraints>"

            GENERAL EXAMPLES for `action_for_next_model` and `prompt_for_next_model`:
            IF USER ASKS TO "generate a python script for X" (and no ImportError mentioned):
                "action_for_next_model": "generate_new_code_m3",
                "prompt_for_next_model": "<RequestDetails>User wants a Python script for X. Ensure comprehensive, runnable code with setup instructions.</RequestDetails><LibraryConstraints>[Extract or infer constraints, e.g., 'Python standard library only' if none mentioned]</LibraryConstraints>"
            IF USER PROVIDES CODE AND SAYS "fix this logical error" or "improve this code" (and no ImportError mentioned):
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
                model_name=self.model_name, safety_settings=safety_settings,
                generation_config=genai_parameters, system_instruction=system_instruction_model1
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
                count = 0
                for msg_data in reversed(ui_chat_history_for_context[:-1]): 
                    if count >= 6: break
                    text_content = ""
                    if "content_parts" in msg_data:
                        for part in msg_data["content_parts"]:
                            if part["type"] == "text": text_content += part["data"] + " "
                            elif part["type"] == "code":
                                text_content += f"\n```python\n{part['data']['code']}\n```\n" # Include code in history context
                    elif "content" in msg_data: text_content = msg_data["content"]
                    if text_content.strip():
                        history_to_send.append(f"{msg_data['role']}: {text_content.strip()}")
                        count += 1
                contextual_prompt_for_model1 += "\n".join(reversed(history_to_send))
            contextual_prompt_for_model1 += f"\n\nBased on the current request and this history, generate your JSON directive."
            
            response = self.chat_session.send_message(contextual_prompt_for_model1)
            raw_text = response.text.strip()

            json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if json_match: json_string = json_match.group(1)
            else: json_string = raw_text
            
            try:
                parsed_json = json.loads(json_string)
                required_keys = ["is_code_related", "user_facing_acknowledgement", 
                                 "action_for_next_model", "prompt_for_next_model", 
                                 "library_constraints_for_next_model"]
                if not all(k in parsed_json for k in required_keys):
                    missing_keys = [k for k in required_keys if k not in parsed_json]
                    print(f"WARNING (make_model1): Model1 JSON missing essential keys: {missing_keys}. Output: {parsed_json}")
                    # Graceful degradation: fill missing keys with defaults
                    parsed_json["is_code_related"] = parsed_json.get("is_code_related", False)
                    parsed_json["user_facing_acknowledgement"] = parsed_json.get("user_facing_acknowledgement", "Sorry, an issue occurred while processing the request structure.")
                    parsed_json["action_for_next_model"] = parsed_json.get("action_for_next_model", None)
                    parsed_json["prompt_for_next_model"] = parsed_json.get("prompt_for_next_model", None)
                    parsed_json["library_constraints_for_next_model"] = parsed_json.get("library_constraints_for_next_model", None)
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
                model_name=self.model_name, safety_settings=self.safety_settings,
                generation_config=self.generation_config, system_instruction=self.system_instruction_text
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

                finish_reason_val = None
                if chunk.candidates and chunk.candidates[0].finish_reason is not None:
                    try:
                        finish_reason_val = chunk.candidates[0].finish_reason.value
                    except AttributeError: 
                        finish_reason_val = chunk.candidates[0].finish_reason 
                
                if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                    reason_message = chunk.prompt_feedback.block_reason_message or "Content blocked by safety filter"
                    print(f"WARNING: Stream from {self.__class__.__name__} blocked. Reason: {reason_message}")
                    yield f"\n\n---STREAM BLOCKED by Safety Filter in {self.__class__.__name__}: {reason_message}---\n"
                    break 
                
                if finish_reason_val is not None:
                    if finish_reason_val == 2: # MAX_TOKENS
                        yield "\n\n---MAX_TOKENS_REACHED---\n"
                        break
                    elif finish_reason_val in [3, 4, 5]: # SAFETY, RECITATION, OTHER
                        reason_map = {3: "SAFETY", 4: "RECITATION", 5: "OTHER"}
                        finish_reason_name = reason_map.get(finish_reason_val, f"UNKNOWN_TERMINAL_REASON_CODE_{finish_reason_val}")
                        yield f"\n\n---STREAM_ENDED_UNEXPECTEDLY ({finish_reason_name})---\n"
                        break
        except Exception as e:
            print(f'ERROR during {self.__class__.__name__} streaming response: {e}')
            import traceback; traceback.print_exc()
            yield f"\n\n--- ERROR in {self.__class__.__name__} while streaming: {e} ---\n\n"


# --- Specialized Model Implementations ---
class make_model2(SpecializedStreamingModel): # Basic Code Generator
    def __init__(self, max_output_tokens=8120, model_name_suffix='latest'):
        system_instruction = """**CORE DIRECTIVE: Elite AI Code Synthesis Engine**

                            **Mission Critical Objective:** Your SOLE function is to synthesize raw, executable, production-grade source code based on the precise specifications provided in the user prompt.

                            **Output Mandate: STRUCTURED CODE OUTPUT**
                            1.  Your response MUST consist of the following, in this order:
                                a.  **SETUP INSTRUCTIONS (Mandatory if external libraries are used):**
                                    *   If the generated code relies on ANY external libraries or packages not part of the standard library for the target language, you MUST provide the necessary installation commands as a comment block at the VERY BEGINNING of your code output.
                                    *   Format:
                                        *   For Python: `# requirements.txt content:` followed by commented lines like `# library1==<version>` or `# pip install library1==<version> library2==<version>`. Specify versions.
                                        *   For Node.js: `// package.json dependencies:` followed by `// npm install library1@<version> library2@<version>`.
                                        *   For other languages, use appropriate comment syntax and package manager commands, clearly indicating library name, install command, and PREFERABLY version.
                                    *   If NO external libraries are used, OMIT this setup block entirely OR provide a comment like `# No external non-standard libraries required.`
                                b.  **PURE SOURCE CODE:**
                                    *   Immediately following the (optional) setup instruction block, provide ONLY the requested source code.
                                    *   **ABSOLUTELY NO** surrounding text, explanations, apologies, conversational remarks, introductions, or summaries are permitted beyond the initial setup instruction comments and the code itself.
                                    *   No markdown code block specifiers (e.g., ```python) are permitted around the setup instructions or the pure source code. Your entire output is the raw text.
                            2.  If the request implies multiple files, structure your output logically (e.g., using clear comment delimiters like `# --- FILENAME: main.py ---` before each file's content) or as directed in the prompt, but still ensure setup instructions appear once at the very top if applicable.

                            **Code Quality Non-Negotiables (Adherence is MANDATORY):**
                                a.  **100% Functional & Correct:** The code MUST compile/interpret and execute flawlessly, producing the correct results for all valid inputs and fulfilling all specified requirements. It must be a complete, working solution.
                                b.  **Maximal Algorithmic Efficiency:**
                                    *   **Time Complexity:** Implement the most efficient algorithms possible. If complex, state and justify Big O in a leading comment block for the relevant function/module (AFTER the global setup instructions).
                                    *   **Space Complexity:** Minimize memory footprint. State space complexity in comments if significant.
                                c.  **Extreme Robustness & Reliability:**
                                    *   Anticipate and gracefully handle ALL conceivable edge cases, invalid inputs, and common operational failures.
                                    *   Implement clear, informative, and actionable error handling.
                                d.  **Security by Design:** If applicable, design with security best practices.
                                e.  **Impeccable Readability & Maintainability (Production Standard):**
                                    *   Strictly adhere to idiomatic style conventions (e.g., PEP 8 for Python).
                                    *   Clear, descriptive naming. Modular design.
                                    *   Concise, high-value comments ONLY for non-obvious logic (explain *why*, not *what*).
                                f.  **Comprehensive & Complete:** Address all requirements.
                                g.  **No Placeholders:** Deliver finished, production-ready code. NO "TODO", "FIXME".

                            **Operational Protocol:**
                            *   You will receive a highly detailed prompt from `model1`.
                            *   Transform these instructions directly into compliant source code following the STRUCTURED CODE OUTPUT mandate.
                            *   Assume the prompt is your complete specification.
                            *   If the prompt specifies a language, you MUST use it.

                            **Performance Standard:** Your output will be judged on its direct usability, adherence to the STRUCTURED CODE OUTPUT format, and the extreme quality standards. Failure to include necessary setup instructions when external libraries are used (with versions), or including any extraneous text, is unacceptable. Synthesize with unparalleled precision.
"""
        super().__init__("make_model2", model_name_suffix, system_instruction, max_output_tokens, temperature=0.2)

class make_model3(SpecializedStreamingModel): # Apex Code Synthesizer
    def __init__(self, max_output_tokens=8120, model_name_suffix='latest'):
        system_instruction = """**CORE DIRECTIVE: Apex AI Code Synthesizer - Master Craftsman of Code**
            **Unwavering Mission:** Your singular, non-negotiable purpose is to transmute highly detailed user specifications, as relayed by an orchestrating AI (model1), into raw, directly executable, production-caliber source code. You are a precision instrument for code generation.
            **Output Protocol: PRECISION-STRUCTURED RAW TEXT CODE**
            Your entire response MUST be raw text, adhering to this exact structure:
            1.  **PART 1: SETUP DIRECTIVES (CONDITIONAL - Precedes Code)**
                *   MANDATORY if external libraries are used. Comment block at ABSOLUTE BEGINNING.
                *   Python Example: `# Required Libraries & Setup:\n#   - numpy: pip install numpy==1.26.4` (SPECIFY EXACT VERSION).
                *   If NO external libraries: `# Standard Library Only - No external setup required.`
            2.  **PART 2: PURE SOURCE CODE (IMMEDIATELY FOLLOWS SETUP)**
                *   ONLY source code. NO markdown wrappers (```python), NO conversational text.
                *   Multi-file: Delimit with `# ===== START FILE: path/to/file.ext =====` etc. Setup is still once at top.
            **Code Quality Mandate: Uncompromising Excellence (Non-Negotiable)**
                a.  **Flawless Execution (1000% Standard):** Must compile/execute flawlessly.
                b.  **Peak Algorithmic & Resource Efficiency:** State Big O for critical parts with justification.
                c.  **Ironclad Robustness & Bulletproof Reliability:** Handle ALL edge cases, invalid inputs, failures.
                d.  **Security by Default & Design:** Secure against common vulnerabilities if applicable.
                e.  **Exemplary Readability & Maintainability (Gold Standard):** PEP 8/equivalent, clear naming, modular, high-impact comments (intent/rationale).
                f.  **Holistic Completeness & Turn-Key Solution.**
                g.  **Zero Placeholders/TODOs/Stubs.**
            **Operational Protocol:**
            *   Input from model1 is sole source of truth. Transform it directly into code.
            *   DO NOT infer beyond prompt unless critical for safety/functionality, and document such inferences in comments.
            **Performance Benchmark:** Judged by immediate fitness for production, adherence to RAW TEXT output, and Uncompromising Excellence. Deviations are critical failures.
"""
        super().__init__("make_model3", model_name_suffix, system_instruction, max_output_tokens, temperature=0.3)

class make_model4(SpecializedStreamingModel): # Grandmaster Code Physician
    def __init__(self, max_output_tokens=8120, model_name_suffix='latest'):
        system_instruction = """**CORE DIRECTIVE: Grandmaster AI Code Physician & Optimization Surgeon**
            **Unyielding Mission:** Diagnose, surgically correct, and optimize source code, adhering to library constraints. Multi-stage mandate: Forensic Analysis -> Strategic Remediation Plan (respecting constraints, checking for versioning issues like deprecated imports) -> Surgical Implementation -> Rigorous Post-Operative Verification -> Comprehensive Reporting & Delivery.
            **Output Mandate: Clinical Two-Part Response (No Deviation Permitted)**
            **PART 1: CLINICAL DIAGNOSIS, TREATMENT & VERIFICATION REPORT (JSON Object)**
            *   Single, valid JSON object in ```json markdown. Schema includes: `clinical_report` with `case_reference_id`, `language_identified_and_version`, `enforced_library_constraints`, `initial_code_prognosis_summary`, `interventions_performed_log` (array with `finding_reference_id`, `original_code_location`, `verbatim_problematic_segment`, `detailed_diagnosis` (should explain if import paths changed due to library versions), `prescribed_treatment_code`, `treatment_rationale_and_constraint_adherence`, `intervention_verification_status`), `outstanding_conditions_due_to_constraints` (array), `final_code_certification_statement` (`certification_status`, `verification_protocol_summary`), `final_code_library_dependencies_and_setup` (with EXACT versions).
            **PART 2: FINAL CERTIFIED SOURCE CODE (Markdown Code Block)**
            *   IMMEDIATELY follows PART 1. ONLY complete, verified, corrected, optimized code, adhering to constraints. Enclosed in language-specific markdown (e.g., ```python ... ```). NO conversational text outside this block.
            **Final Code Standards:** 1000% Error-Free & Runnable (within constraints). Optimized Efficiency (within constraints). Unyielding Library Constraint Adherence. Exemplary Robustness, Readability.
            **Operational Protocol:** Input: `<CodeToFix>` & `<RequestDetails>` (with constraints & error context if any). Library constraints are ABSOLUTE. If an import error is noted, specifically investigate if the item was moved/deprecated in newer library versions and provide the modern, correct import.
            **Performance Benchmark:** Judged on PART 2's quality/runnability/constraint adherence, and PART 1's accuracy (especially for version-aware import fixes). Violating constraints is critical failure.
"""
        super().__init__("make_model4", model_name_suffix, system_instruction, max_output_tokens, temperature=0.4)

class make_model5(SpecializedStreamingModel): # Iterative Self-Correcting Refiner
    def __init__(self, max_output_tokens=8120, model_name_suffix='latest'):
        system_instruction = """**CORE DIRECTIVE: Autonomous AI Code Resilience & Perfection Engine**
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
"""
        super().__init__("make_model5", model_name_suffix, system_instruction, max_output_tokens, temperature=0.5)

class make_model_ml_optimizer(SpecializedStreamingModel): # ML Performance Optimizer
    def __init__(self, max_output_tokens=8192, model_name_suffix='latest'):
        system_instruction = """**CORE DIRECTIVE: AI Peak Performance ML Engineering Specialist**
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
"""
        super().__init__("make_model_ml_optimizer", model_name_suffix, system_instruction, max_output_tokens, temperature=0.4)
