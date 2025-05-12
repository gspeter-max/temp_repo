import streamlit as st
import json
import re
import time # Keep for any explicit delays or for generating unique IDs if needed
from model1 import make_model1, make_model2, make_model3, make_model4, make_model5

# --- Page Configuration ---
st.set_page_config(page_title="AI Super Assistant Pro", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS for ChatGPT-like appearance ---
st.markdown("""
    <style>
    /* ... (keep your existing CSS or enhance it) ... */
    .stChatMessageContent code { /* Style for inline code if markdown renders it */
        font-size: 0.9em;
        padding: 0.1em 0.3em;
        background-color: #272822; /* Monokai-ish background */
        border-radius: 3px;
        color: #f8f8f2; /* Light text for dark background */
    }
    .stCodeBlock > div { /* Target the inner div of st.code for better styling */
        border: 1px solid #555;
        border-radius: 8px;
    }
    /* Ensure chat input is at the bottom */
    .main .stChatInputContainer {
        position: fixed;
        bottom: 0;
        width: calc(100% - 2rem); /* Adjust based on your app's padding */
        background-color: #0e1117; /* Match Streamlit dark theme background */
        padding: 1rem;
        z-index: 99;
    }
    /* Adjust main content padding to prevent overlap with fixed chat input */
     .main > div:first-child {
        padding-bottom: 6rem; /* Height of your chat input area + some buffer */
    }
    </style>
""", unsafe_allow_html=True)

# --- Backend Model Initialization ---
if 'models_initialized' not in st.session_state:
    st.session_state.models_initialized = False
if not st.session_state.models_initialized:
    with st.spinner("Initializing AI Cores... Please wait."):
        try:
            st.session_state.model1_instance = make_model1()
            st.session_state.model2_instance = make_model2()
            st.session_state.model3_instance = make_model3()
            st.session_state.model4_instance = make_model4()
            st.session_state.model5_instance = make_model5()
            st.session_state.models_initialized = True
            print("INFO: All AI models re-initialized successfully for Streamlit.")
        except Exception as e:
            st.error(f"Fatal Error: Could not initialize AI models: {e}")
            # Log full traceback to console for debugging
            import traceback
            traceback.print_exc()
            st.stop()

# --- Helper Function to Parse and Display AI's Multi-Part Streamed or Full Response ---
def display_ai_parts_from_string(full_response_string, container):
    """
    Parses a full string that might contain a JSON block then a Markdown code block,
    or raw code (Model3 style), and displays them.
    """
    if not full_response_string or not full_response_string.strip():
        # container.markdown("*AI provided no further textual output for this part.*")
        return

    # Priority 1: Model4/Model5 style (JSON report in ```json, then code in ```language)
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", full_response_string, re.DOTALL)
    remaining_after_json = full_response_string

    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            container.json(json_data)
            remaining_after_json = full_response_string.replace(json_match.group(0), "", 1).strip()
        except json.JSONDecodeError as e:
            container.warning(f"AI Warning: Could not parse a JSON block: {e}. Displaying as text.")
            # Fall through, remaining_after_json is still the full string

    # Check remaining text for a markdown code block
    code_match = re.search(r"```(\w*)\s*\n(.*?)\n```", remaining_after_json, re.DOTALL)
    remaining_after_code = remaining_after_json

    if code_match:
        language = code_match.group(1).lower() if code_match.group(1) else "plaintext"
        code_content = code_match.group(2)
        container.code(code_content, language=language)
        remaining_after_code = remaining_after_json.replace(code_match.group(0), "", 1).strip()
    
    # If no JSON and no Markdown code block was found using regex above,
    # it might be Model3 style (raw setup comments + raw code) or simple text.
    elif not json_match and remaining_after_json.strip(): # No JSON, but there's text
        # Heuristic for Model3 style: starts with a comment or typical code keywords
        text_to_check = remaining_after_json.strip()
        if text_to_check.startswith(("#", "//")) or \
           any(kw in text_to_check for kw in ["def ", "class ", "import ", "function ", "const ", "let "]):
            # It looks like raw code (potentially with setup comments)
            container.code(text_to_check, language="python") # Default to python, or try to infer
            remaining_after_code = "" # Assume all of it was code
        else: # It's likely just plain markdown text
            pass # Will be handled by the final markdown display

    # Display any text that remains after extracting JSON and/or markdown-wrapped code
    if remaining_after_code.strip():
        container.markdown(remaining_after_code)


# --- Streamlit UI ---
st.title("🚀 AI Super Assistant Pro")
# st.caption("Your advanced AI partner for coding, ML, and more. Powered by Gemini.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content_parts": [{"type": "text", "data": "Hello! How can I help you with coding, ML, or general queries today?"}]}
    ]

# Display chat messages
# We now store content_parts to handle mixed content better when re-rendering
for msg_idx, message_data in enumerate(st.session_state.messages):
    with st.chat_message(message_data["role"]):
        if "content_parts" in message_data:
            for part in message_data["content_parts"]:
                if part["type"] == "text":
                    st.markdown(part["data"])
                elif part["type"] == "json":
                    st.json(part["data"])
                elif part["type"] == "code":
                    st.code(part["data"]["code"], language=part["data"]["language"])
        else: # Fallback for older simple string content
            st.markdown(message_data["content"])


# --- Main Chat Input Logic ---
if user_input := st.chat_input("Ask me anything..."):
    if not st.session_state.models_initialized:
        st.error("Models are not initialized. Please check the console for errors.")
        st.stop()

    # Add user message to UI history
    st.session_state.messages.append({"role": "user", "content_parts": [{"type": "text", "data": user_input}]})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant's turn
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🧠 Thinking...")

        assistant_response_parts = [] # To store structured parts of the response

        try:
            # 1. Call Model1 (Orchestrator)
            # Prepare history for Model1 (it uses its own internal chat_session, but might need initial context)
            # For simplicity, model1's __call__ here takes current prompt, and manages its own history.
            model1_output_dict = st.session_state.model1_instance(user_input) # Model1's __call__ now returns a dict

            if not isinstance(model1_output_dict, dict):
                raise ValueError(f"Model1 did not return a dictionary. Received: {type(model1_output_dict)}")

            is_code_related = model1_output_dict.get("is_code_related", False)
            user_ack_from_model1 = model1_output_dict.get("response_for_user", "")
            action_for_next = model1_output_dict.get("action_for_next_model")
            prompt_for_next = model1_output_dict.get("prompt_for_next_model")

            if user_ack_from_model1: # Display M1's direct response/ack first
                assistant_response_parts.append({"type": "text", "data": user_ack_from_model1})
                thinking_placeholder.markdown(user_ack_from_model1 + (" Processing further..." if is_code_related else ""))
            
            current_code_state = "" # To hold output from M2/M3 for M4/M5

            if is_code_related and action_for_next and prompt_for_next:
                specialized_model_output_accumulator = []

                # Determine which model to call based on model1's action
                # This is a simplified pipeline dispatch. You can make it more complex.
                
                # --- PIPELINE EXECUTION ---
                # For a complex pipeline, you might want to show progress for each stage
                
                # Stage: model2 (Initial Code Generation)
                if action_for_next == "generate_new_code_m2_style": # Assuming M1 directs to an M2 style generator
                    thinking_placeholder.markdown(user_ack_from_model1 + " (Stage 2: Generating initial code draft...)")
                    for chunk in st.session_state.model2_instance(prompt_for_next):
                        specialized_model_output_accumulator.append(chunk)
                        thinking_placeholder.markdown(user_ack_from_model1 + " (Stage 2: Generating initial code draft...)\n```\n" + "".join(specialized_model_output_accumulator) + "...\n```")
                    current_code_state = "".join(specialized_model_output_accumulator)
                    # For now, assume M2's raw output is the primary deliverable if it's the only one called by M1
                    # If M2 is just the first step of a longer chain, we might not add its direct output to assistant_response_parts yet.
                    # For this example, let's say M2's output IS the final if M1 specified this action.
                    assistant_response_parts.append({"type": "raw_code_model3_style", "data": current_code_state})


                # Stage: model3 (Apex Synthesizer - could take M1 prompt or M2 output)
                elif action_for_next == "generate_apex_code_m3":
                    thinking_placeholder.markdown(user_ack_from_model1 + " (Stage 2/X: Synthesizing Apex code...)")
                    for chunk in st.session_state.model3_instance(prompt_for_next): # M3 takes M1's detailed prompt
                        specialized_model_output_accumulator.append(chunk)
                        thinking_placeholder.markdown(user_ack_from_model1 + " (Stage 2/X: Synthesizing Apex code... Current output constructing...)")
                    current_code_state = "".join(specialized_model_output_accumulator)
                    assistant_response_parts.append({"type": "raw_code_model3_style", "data": current_code_state}) # Model3 outputs raw code

                # Stage: model4 (Code Physician - takes code, outputs JSON report + MD code)
                elif action_for_next == "fix_and_verify_code_m4":
                    # prompt_for_next from model1 should be the <CodeToFix>...</CodeToFix><RequestDetails>...</RequestDetails>
                    thinking_placeholder.markdown(user_ack_from_model1 + " (Stage 2/X: Diagnosing & Correcting Code with Model4...)")
                    for chunk in st.session_state.model4_instance(prompt_for_next):
                        specialized_model_output_accumulator.append(chunk)
                        thinking_placeholder.markdown(user_ack_from_model1 + " (Stage 2/X: Diagnosing & Correcting... Report/Code generating...)")
                    current_code_state = "".join(specialized_model_output_accumulator) # This will be JSON + MD Code
                    assistant_response_parts.append({"type": "model4_5_output_style", "data": current_code_state})

                # Stage: model5 (Iterative Refiner - takes code, outputs JSON log + MD code)
                elif action_for_next == "iteratively_perfect_code_m5":
                    thinking_placeholder.markdown(user_ack_from_model1 + " (Stage 2/X: Iteratively Perfecting Code with Model5...)")
                    for chunk in st.session_state.model5_instance(prompt_for_next):
                        specialized_model_output_accumulator.append(chunk)
                        thinking_placeholder.markdown(user_ack_from_model1 + " (Stage 2/X: Iteratively Perfecting... Log/Code generating...)")
                    current_code_state = "".join(specialized_model_output_accumulator)
                    assistant_response_parts.append({"type": "model4_5_output_style", "data": current_code_state})
                
                # Add more elif for other actions from model1 if you expand your pipeline
                # e.g., for the ML Optimizer model.

                else: # Default if action specified but not handled above
                    if not user_ack_from_model1: # If no ack from M1 for this simple text response
                         assistant_response_parts.append({"type": "text", "data": "I've received your request. I'm not yet fully equipped for that specific complex action in this demo."})


            elif not is_code_related and not user_ack_from_model1 : # Model1 had no direct response_for_user but it's not code related
                 assistant_response_parts.append({"type": "text", "data": "I'm here to help. What would you like to do?"})


            thinking_placeholder.empty() # Clear "Thinking..." or progress messages

            # Display the accumulated and structured parts
            for part_idx, part_data in enumerate(assistant_response_parts):
                if part_idx > 0: st.markdown("---") # Separator if multiple parts from M1 + M_specialized
                if part_data["type"] == "text":
                    st.markdown(part_data["data"])
                elif part_data["type"] == "raw_code_model3_style":
                    # Model3 output is setup_comments + raw_code
                    st.code(part_data["data"], language="python") # Assuming python, or try to infer
                elif part_data["type"] == "model4_5_output_style":
                    # This is a string containing JSON block then MD code block
                    display_ai_parts_from_string(part_data["data"], st)


        except Exception as e:
            thinking_placeholder.error(f"An error occurred in the AI pipeline: {e}")
            assistant_response_parts = [{"type": "text", "data": f"Sorry, I encountered an error: {e}"}]
            # Log full traceback to console for debugging
            import traceback
            traceback.print_exc()
            st.markdown(f"Error: {e}") # Display error in chat too

    # Add final assistant response parts to session state for re-rendering
    if assistant_response_parts:
        st.session_state.messages.append({"role": "assistant", "content_parts": assistant_response_parts})
    elif not is_code_related and user_ack_from_model1 and not assistant_response_parts: # M1 gave simple response, no further action
        st.session_state.messages.append({"role": "assistant", "content_parts": [{"type":"text", "data":user_ack_from_model1}]})
