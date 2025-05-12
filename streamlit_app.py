import streamlit as st
import json
import re
import time
# Import your model classes from model1.py
from model1 import make_model1, make_model2, make_model3, make_model4, make_model5, make_model_ml_optimizer

# --- Page Configuration ---
st.set_page_config(page_title="AI Super Assistant Pro", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS (Keep or enhance your existing CSS) ---
st.markdown("""
    <style>
    /* Your existing or new CSS styles here */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="stChatMessageContentUser"] {
        background-color: #2b313e;
    }
    .stChatMessage[data-testid="stChatMessageContentAssistant"] {
        background-color: #40414f;
    }
    .stCodeBlock > div {
        border: 1px solid #555;
        border-radius: 8px;
    }
    .main .stChatInputContainer {
        position: fixed;
        bottom: 0;
        width: calc(100% - 2rem); /* Adjust based on your app's padding */
        background-color: #0e1117; /* Match Streamlit dark theme background */
        padding: 0.5rem 1rem; /* Adjusted padding */
        z-index: 99;
        border-top: 1px solid #333;
    }
    .main > div:first-child { /* Main content area */
        padding-bottom: 6rem; /* Space for the fixed chat input */
    }
    </style>
""", unsafe_allow_html=True)

# --- Model Initialization (once per session) ---
if 'models_initialized' not in st.session_state:
    st.session_state.models_initialized = False

if not st.session_state.models_initialized:
    with st.spinner("Initializing AI Cores... This might take a moment."):
        try:
            # It's good practice to wrap individual initializations too
            st.session_state.model1_instance = make_model1()
            st.session_state.model2_instance = make_model2() # If you still use it directly
            st.session_state.model3_instance = make_model3()
            st.session_state.model4_instance = make_model4()
            st.session_state.model5_instance = make_model5()
            st.session_state.model_ml_optimizer_instance = make_model_ml_optimizer()
            st.session_state.models_initialized = True
            print("INFO: All AI models initialized successfully for Streamlit session.")
        except RuntimeError as e: # Catch the specific error from AIModelBase
            st.error(f"Fatal Error during AI Model Initialization: {e}")
            st.error("Please ensure your GOOGLE_API_KEY is correctly set in 'api_key.env' and the file is in the same directory as model1.py.")
            st.session_state.models_initialized = False # Explicitly set to false
            st.stop() # Halt app execution
        except Exception as e:
            st.error(f"An unexpected fatal error occurred during AI Model Initialization: {e}")
            import traceback
            traceback.print_exc()
            st.session_state.models_initialized = False
            st.stop()


# --- Helper Function to Parse and Display AI's Multi-Part Response ---
def display_ai_parts_from_string(full_response_string, container):
    """
    Parses a full string that might contain a JSON block then a Markdown code block (Model4/5/ML style),
    or raw code with setup comments (Model3 style), and displays them.
    """
    if not full_response_string or not full_response_string.strip():
        return [{"type": "text", "data": "*AI provided no further textual output for this part.*"}]

    parts_to_store_and_display = []
    remaining_text = full_response_string

    # Priority 1: Model4/Model5/ML_Optimizer style (JSON report in ```json, then code in ```language)
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", remaining_text, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            parts_to_store_and_display.append({"type": "json", "data": json_data})
            container.json(json_data)
            remaining_text = remaining_text.replace(json_match.group(0), "", 1).strip()
        except json.JSONDecodeError as e:
            container.warning(f"AI Warning: Could not parse a JSON block: {e}. Displaying as text.")
            # The text will be handled by subsequent checks or as final markdown

    # Check remaining text for a markdown code block
    code_match = re.search(r"```(\w*)\s*\n(.*?)\n```", remaining_text, re.DOTALL)
    if code_match:
        language = code_match.group(1).lower() if code_match.group(1) else "plaintext"
        code_content = code_match.group(2).strip() # Strip to remove surrounding newlines from regex capture
        parts_to_store_and_display.append({"type": "code", "data": {"language": language, "code": code_content}})
        container.code(code_content, language=language)
        remaining_text = remaining_text.replace(code_match.group(0), "", 1).strip()
    
    # If no JSON and no Markdown code block was found using regex above,
    # it might be Model3 style (raw setup comments + raw code) or simple text.
    elif not json_match and remaining_text.strip():
        text_to_check = remaining_text.strip()
        # Heuristic for Model3 style: starts with a comment or typical code keywords
        # AND doesn't look like it's just a fragment of a sentence.
        is_likely_raw_code = text_to_check.startswith(("#", "//")) or \
                             any(kw in text_to_check for kw in ["def ", "class ", "import ", "function ", "const ", "let "])
        
        if is_likely_raw_code:
            # Try to infer language, default to python. A more robust way would be if model1 signals model3 was used.
            lang_guess = "python" # Basic guess
            if "function " in text_to_check and "{" in text_to_check: lang_guess = "javascript"
            elif "public class" in text_to_check: lang_guess = "java"
            
            parts_to_store_and_display.append({"type": "code", "data": {"language": lang_guess, "code": text_to_check}})
            container.code(text_to_check, language=lang_guess)
            remaining_text = "" # Assume all of it was code
    
    # Display any text that remains after extracting JSON and/or markdown-wrapped/raw code
    if remaining_text.strip():
        parts_to_store_and_display.append({"type": "text", "data": remaining_text})
        container.markdown(remaining_text)
        
    return parts_to_store_and_display


# --- Streamlit UI ---
st.title("🚀 AI Super Assistant Pro")

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content_parts": [{"type": "text", "data": "Hello! I'm your AI Super Assistant. How can I help with your coding or ML projects today?"}]}
    ]

# Display chat messages from history
# This ensures that when Streamlit reruns, the history is displayed correctly
for msg_data in st.session_state.messages:
    with st.chat_message(msg_data["role"]):
        if "content_parts" in msg_data:
            for part in msg_data["content_parts"]:
                if part["type"] == "text":
                    st.markdown(part["data"])
                elif part["type"] == "json":
                    st.json(part["data"])
                elif part["type"] == "code":
                    st.code(part["data"]["code"], language=part["data"]["language"])
        elif "content" in msg_data: # Fallback for old simple string content, less ideal
            st.markdown(msg_data["content"])


# --- Main Chat Input Logic ---
if user_input := st.chat_input("What can I craft or fix for you today?"):
    if not st.session_state.get("models_initialized", False): # Check before proceeding
        st.error("AI Models are not ready. Please check the initial startup messages or console logs.")
        st.stop()

    # Add user message to UI history and display it
    st.session_state.messages.append({"role": "user", "content_parts": [{"type": "text", "data": user_input}]})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant's turn
    with st.chat_message("assistant"):
        progress_bar_container = st.empty() # For "Thinking..." and streaming progress
        # Use a single container for all assistant output parts in this turn
        # This ensures all parts (ack, json, code) appear within the same assistant chat bubble
        current_turn_assistant_output_container = st.container()


        assistant_response_accumulator = [] # To build the full string for specialized models
        final_structured_parts_for_history = [] # To store for st.session_state.messages

        try:
            progress_bar_container.markdown("🧠 Accessing Orchestrator (Model 1)...")
            # 1. Call Model1 (Orchestrator)
            # Model1 uses its internal chat_session for its own context if needed.
            # We pass the Streamlit UI history for broader context if Model1 is designed to use it.
            model1_output_dict = st.session_state.model1_instance(user_input, st.session_state.messages)

            if not isinstance(model1_output_dict, dict):
                raise ValueError(f"Model1 did not return a dictionary. Received: {type(model1_output_dict)}. Output: {model1_output_dict}")

            is_code_related = model1_output_dict.get("is_code_related", False)
            user_ack_from_model1 = model1_output_dict.get("user_facing_acknowledgement", "")
            action_for_next = model1_output_dict.get("action_for_next_model")
            prompt_for_next = model1_output_dict.get("prompt_for_next_model")
            # lib_constraints = model1_output_dict.get("library_constraints_for_next_model") # Use this in prompts

            if user_ack_from_model1:
                current_turn_assistant_output_container.markdown(user_ack_from_model1)
                final_structured_parts_for_history.append({"type": "text", "data": user_ack_from_model1})
                if is_code_related:
                    progress_bar_container.markdown(user_ack_from_model1 + " Engaging specialized AI...")
                else:
                    progress_bar_container.empty() # Clear thinking message if it's just chat

            if is_code_related and action_for_next and prompt_for_next:
                # Map action to model instance and expected output style
                model_map = {
                    "generate_new_code_m3": (st.session_state.model3_instance, "raw_code_model3_style", "Synthesizing Code (Model 3)..."),
                    "fix_and_verify_code_m4": (st.session_state.model4_instance, "model4_5_output_style", "Diagnosing & Fixing (Model 4)..."),
                    "iteratively_perfect_code_m5": (st.session_state.model5_instance, "model4_5_output_style", "Iteratively Perfecting (Model 5)..."),
                    "optimize_ml_solution_m_ml": (st.session_state.model_ml_optimizer_instance, "model4_5_output_style", "Optimizing ML Solution...")
                    # Add your "model2" action here if it's distinct from model3
                    # "generate_code_m2_simple": (st.session_state.model2_instance, "raw_code_model3_style", "Generating Draft (Model 2)...")
                }

                if action_for_next in model_map:
                    target_model_instance, output_style, progress_message = model_map[action_for_next]
                    
                    progress_bar_container.markdown(progress_message)
                    
                    # Temporary placeholder for the streaming content within the current assistant message
                    stream_display_placeholder = current_turn_assistant_output_container.empty()
                    current_stream_text = ""

                    for chunk in target_model_instance(prompt_for_next): # Call the specialized model
                        assistant_response_accumulator.append(chunk)
                        current_stream_text = "".join(assistant_response_accumulator)
                        # Update placeholder with accumulating stream
                        # For raw code, st.code is better; for mixed, markdown is okay for progress
                        if output_style == "raw_code_model3_style":
                            stream_display_placeholder.code(current_stream_text + " ▌", language="python") # Simulate cursor
                        else: # For JSON+Code, markdown is fine for streaming progress
                            stream_display_placeholder.markdown(current_stream_text + " ▌")
                    
                    stream_display_placeholder.empty() # Clear the streaming progress placeholder
                    
                    final_output_string = "".join(assistant_response_accumulator)
                    
                    # Now parse and display the complete final output from the specialized model
                    # using the main container for this assistant turn
                    parsed_parts = display_ai_parts_from_string(final_output_string, current_turn_assistant_output_container)
                    final_structured_parts_for_history.extend(parsed_parts)

                else:
                    msg = f"Orchestrator (Model 1) requested an unknown action: '{action_for_next}'."
                    current_turn_assistant_output_container.warning(msg)
                    final_structured_parts_for_history.append({"type": "text", "data": msg})

            elif not is_code_related and not user_ack_from_model1:
                # Model1 decided it's not code related but didn't provide a specific user_facing_acknowledgement
                # This can happen if Model1's JSON output for "response_for_user" was empty.
                fallback_chat_msg = "I'm ready to help. What's on your mind?"
                current_turn_assistant_output_container.markdown(fallback_chat_msg)
                final_structured_parts_for_history.append({"type": "text", "data": fallback_chat_msg})

            progress_bar_container.empty() # Clear any final progress/thinking message

        except Exception as e:
            progress_bar_container.empty()
            error_msg = f"An error occurred in the AI pipeline: {e}"
            current_turn_assistant_output_container.error(error_msg)
            final_structured_parts_for_history.append({"type": "text", "data": f"Sorry, I encountered an error: {e}"})
            import traceback
            traceback.print_exc() # To console for dev

    # Add final assistant response (as structured parts) to session state for durable display
    if final_structured_parts_for_history:
        st.session_state.messages.append({"role": "assistant", "content_parts": final_structured_parts_for_history})
    elif not final_structured_parts_for_history and user_ack_from_model1 and not is_code_related:
        # This handles the case where Model1 provides a simple chat response and no further action.
        # The ack was already displayed, now store it.
        st.session_state.messages.append({"role": "assistant", "content_parts": [{"type": "text", "data": user_ack_from_model1}]})
