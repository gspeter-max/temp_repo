import streamlit as st
import json
import re
import time
from model1 import (
    make_model1,
    # make_model2, # Decide if M2 is used directly or if M3 always supersedes it
    make_model3,
    make_model4,
    make_model5,
    make_model_ml_optimizer
)

# --- Page Configuration ---
st.set_page_config(
    page_title="GenAI Super Coder Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        body, .main { color: #E0E0E0; background-color: #0E1117; }
        .main > div:first-child { padding-bottom: 8rem !important; }
        .stChatMessage { border-radius: 12px; padding: 0.8rem 1.0rem; margin-bottom: 1rem;
                         border: 1px solid #303238; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                         word-wrap: break-word; overflow-wrap: break-word; }
        .stChatMessage[data-testid="stChatMessageContentUser"] {
            background-color: #2A3942; color: #FFFFFF; margin-left: auto;
            max-width: 70%; float: right; clear: both; }
        .stChatMessage[data-testid="stChatMessageContentUser"] p { color: #FFFFFF !important; }
        .stChatMessage[data-testid="stChatMessageContentAssistant"] {
            background-color: #2F3136; color: #DCDDDE; margin-right: auto;
            max-width: 85%; float: left; clear: both; }
        .stChatMessage[data-testid="stChatMessageContentAssistant"] p { color: #DCDDDE !important; }
        .stChatMessage[data-testid="stChatMessageContentAssistant"] .stCodeBlock,
        .stChatMessage[data-testid="stChatMessageContentAssistant"] .stJson {
             margin-top: 0.5rem; margin-bottom: 0.5rem; }
        .stCodeBlock { border-radius: 8px !important; border: 1px solid #40444B !important;
                       background-color: #1E1F22 !important; }
        .stCodeBlock pre { background-color: #1E1F22 !important; padding: 0.75rem !important; }
        .stCodeBlock pre code { color: #DCDDDE !important; background-color: transparent !important; }
        .stJson { border-radius: 8px; border: 1px solid #40444B; padding: 0.75rem;
                  background-color: #1E1F22 !important; color: #DCDDDE !important; }
        .stJson pre { color: #DCDDDE !important; background-color: transparent !important; }
        .main .stChatInputContainer { position: fixed; bottom: 0; left: 0; right: 0; width: 100%;
                                     background-color: #1A1B1E; padding: 0.6rem 1.2rem;
                                     border-top: 1px solid #303238; box-shadow: 0 -1px 5px rgba(0,0,0,0.15);
                                     z-index: 999; }
        .stTextInput > div > div > input { background-color: #2C2E33 !important; color: #E0E0E0 !important;
                                           border: 1px solid #40444B !important; border-radius: 8px !important;
                                           padding: 0.5rem 0.75rem !important; }
        .stTextInput > div > div > input::placeholder { color: #72767D !important; }
        .stChatInputContainer button { background-color: #4F545C !important; color: white !important;
                                      border-radius: 8px !important; border: none !important; }
        .stChatInputContainer button:hover { background-color: #5D6169 !important; }
        .thinking-placeholder p { font-style: italic; color: #8A8F98; padding: 0.5rem 0; }
    </style>
""", unsafe_allow_html=True)

# --- Model Initialization ---
if 'models_initialized_flag' not in st.session_state: st.session_state.models_initialized_flag = False
if not st.session_state.models_initialized_flag:
    with st.spinner("Initializing AI Cores... This might take a moment for the first time."): # CORRECTED
        try:
            st.session_state.model1_instance = make_model1()
            # st.session_state.model2_instance = make_model2() # Only if M1 routes to it
            st.session_state.model3_instance = make_model3()
            st.session_state.model4_instance = make_model4()
            st.session_state.model5_instance = make_model5()
            st.session_state.model_ml_optimizer_instance = make_model_ml_optimizer()
            st.session_state.models_initialized_flag = True
            print("INFO (Streamlit): All AI models initialized successfully.")
        except RuntimeError as e:
            st.error(f"Fatal Error during AI Model Initialization: {e}")
            st.error("Ensure GOOGLE_API_KEY is in 'api_key.env' (same dir as model1.py) or set as environment variable.")
            st.session_state.models_initialized_flag = False
        except Exception as e:
            st.error(f"An unexpected fatal error during AI Model Initialization: {e}")
            import traceback; traceback.print_exc()
            st.session_state.models_initialized_flag = False

# --- Helper Function to Parse and Display AI's Multi-Part Response ---
def display_ai_parts_from_string(full_response_string, container_to_write_in):
    # ... (The display_ai_parts_from_string function from the previous "Full Code" response - it was already quite robust) ...
    # Key parts:
    # - Parses for ```json block first, displays with container.json()
    # - Parses for ```language block next, displays with container.code()
    # - Checks for Model3-style raw code (setup comments + code), displays with container.code()
    # - Displays any remaining text with container.markdown()
    # - Returns a list of structured parts for history storage.
    # For brevity, I'm not repeating the exact same code here, but ensure you have the robust version.
    if not full_response_string or not full_response_string.strip():
        return [{"type": "text", "data": "*AI provided no output or only whitespace for this part.*"}]

    displayed_parts_for_history = []
    remaining_text = full_response_string.strip() 

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", remaining_text, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            displayed_parts_for_history.append({"type": "json", "data": json_data})
            container_to_write_in.json(json_data)
            remaining_text = remaining_text.replace(json_match.group(0), "", 1).strip()
        except json.JSONDecodeError as e:
            container_to_write_in.warning(f"AI Warning: Could not parse JSON block: {e}.")

    code_match_md = re.search(r"```(\w*)\s*\n(.*?)\n```", remaining_text, re.DOTALL)
    if code_match_md:
        language = code_match_md.group(1).lower().strip() if code_match_md.group(1) else "plaintext"
        code_content = code_match_md.group(2).strip()
        displayed_parts_for_history.append({"type": "code", "data": {"language": language, "code": code_content}})
        container_to_write_in.code(code_content, language=language)
        remaining_text = remaining_text.replace(code_match_md.group(0), "", 1).strip()
    
    elif not json_match and remaining_text: 
        is_likely_raw_code = remaining_text.startswith(("# Required Libraries & Setup:", "// Required Libraries & Setup:", "# Standard Library Only")) or \
                             any(kw in remaining_text for kw in ["def ", "class ", "import ", "function ", "const ", "let "])
        if is_likely_raw_code:
            lang_guess = "python"
            if "function " in remaining_text and "{" in remaining_text and not remaining_text.strip().startswith("def "): lang_guess = "javascript"
            elif "public class" in remaining_text and "{" in remaining_text: lang_guess = "java"
            
            displayed_parts_for_history.append({"type": "code", "data": {"language": lang_guess, "code": remaining_text}})
            container_to_write_in.code(remaining_text, language=lang_guess)
            remaining_text = ""

    if remaining_text.strip():
        displayed_parts_for_history.append({"type": "text", "data": remaining_text})
        container_to_write_in.markdown(remaining_text)
        
    if not displayed_parts_for_history and (full_response_string and full_response_string.strip()):
         displayed_parts_for_history.append({"type": "text", "data": full_response_string}) # Fallback
    return displayed_parts_for_history

# --- Streamlit UI Title ---
st.title("✨ GenAI Super Coder ✨")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content_parts": [{"type": "text", "data": "Hello! I'm your AI Super Coder. How can I assist with your coding or machine learning projects today?"}]}
    ]

for msg_data in st.session_state.messages:
    with st.chat_message(msg_data["role"]):
        if "content_parts" in msg_data:
            for part_idx, part in enumerate(msg_data["content_parts"]):
                if part_idx > 0 and msg_data["role"] == "assistant": st.markdown("---") 
                if part["type"] == "text": st.markdown(part["data"])
                elif part["type"] == "json": st.json(part["data"])
                elif part["type"] == "code": st.code(part["data"]["code"], language=part["data"]["language"])
        elif "content" in msg_data: st.markdown(msg_data["content"])


if user_input := st.chat_input("Describe your coding task or ask a question..."):
    if not st.session_state.get("models_initialized_flag", False):
        st.error("AI Models are not ready. Please check startup messages or console logs.")
    else:
        st.session_state.messages.append({"role": "user", "content_parts": [{"type": "text", "data": user_input}]})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            current_assistant_turn_container = st.container()
            thinking_placeholder = current_assistant_turn_container.empty()
            thinking_placeholder.markdown("<p class='thinking-placeholder'>🧠 Orchestrating AI response...</p>", unsafe_allow_html=True)

            accumulated_final_parts_for_history = []

            try:
                model1_output_dict = st.session_state.model1_instance(user_input, st.session_state.messages)

                if not isinstance(model1_output_dict, dict):
                    raise ValueError(f"Model1 (Orchestrator) did not return a dictionary. Received: {type(model1_output_dict)}. Output: {model1_output_dict}")

                is_code_related = model1_output_dict.get("is_code_related", False)
                user_ack_from_model1 = model1_output_dict.get("user_facing_acknowledgement", "")
                action_for_next = model1_output_dict.get("action_for_next_model")
                prompt_for_next = model1_output_dict.get("prompt_for_next_model")

                initial_ack_displayed_in_turn = False
                if user_ack_from_model1:
                    if len(user_ack_from_model1.strip()) > 3 :
                        accumulated_final_parts_for_history.append({"type": "text", "data": user_ack_from_model1})
                        current_assistant_turn_container.markdown(user_ack_from_model1)
                        initial_ack_displayed_in_turn = True
                    
                    thinking_placeholder_text = user_ack_from_model1 if initial_ack_displayed_in_turn else "Processing..."
                    if is_code_related:
                        thinking_placeholder.markdown(f"<p class='thinking-placeholder'>{thinking_placeholder_text} Engaging specialized AI...</p>", unsafe_allow_html=True)
                    else:
                        thinking_placeholder.empty()

                if is_code_related and action_for_next and prompt_for_next:
                    model_map = {
                        "generate_new_code_m3": (st.session_state.model3_instance, "Synthesizing High-Quality Code (Model 3)..."),
                        "fix_and_verify_code_m4": (st.session_state.model4_instance, "Diagnosing & Correcting Code (Model 4)..."),
                        "iteratively_perfect_code_m5": (st.session_state.model5_instance, "Iteratively Perfecting Code (Model 5)..."),
                        "optimize_ml_solution_m_ml": (st.session_state.model_ml_optimizer_instance, "Engineering Optimal ML Solution..."),
                    }

                    if action_for_next in model_map:
                        target_model_instance, progress_message_template = model_map[action_for_next]
                        
                        stream_accumulator = []
                        stream_display_area_in_container = current_assistant_turn_container.empty() 

                        thinking_placeholder.markdown(f"<p class='thinking-placeholder'>{progress_message_template} Streaming output...</p>", unsafe_allow_html=True)
                        
                        for chunk_idx, chunk_text in enumerate(target_model_instance(prompt_for_next)):
                            stream_accumulator.append(chunk_text)
                            if chunk_idx % 3 == 0 or len(chunk_text) > 10:
                                stream_display_area_in_container.markdown("".join(stream_accumulator) + " ▌")
                        
                        stream_display_area_in_container.empty()
                        thinking_placeholder.empty()
                        
                        final_output_string_from_specialized_model = "".join(stream_accumulator)
                        
                        parsed_parts = display_ai_parts_from_string(final_output_string_from_specialized_model, current_assistant_turn_container)
                        accumulated_final_parts_for_history.extend(parsed_parts)
                    else:
                        msg = f"Orchestrator (Model 1) requested an unhandled action: '{action_for_next}'."
                        current_assistant_turn_container.warning(msg)
                        accumulated_final_parts_for_history.append({"type": "text", "data": msg})
                
                elif not is_code_related and not user_ack_from_model1:
                    fallback_msg = "I'm ready to assist. What can I do for you?"
                    current_assistant_turn_container.markdown(fallback_msg)
                    accumulated_final_parts_for_history.append({"type": "text", "data": fallback_msg})
                
                if not is_code_related and user_ack_from_model1 and not action_for_next and not initial_ack_displayed_in_turn:
                     # This handles if M1 gave a simple ack that wasn't "substantial" enough to be displayed above,
                     # but it was the only intended output for this turn.
                     accumulated_final_parts_for_history.append({"type": "text", "data": user_ack_from_model1})
                     current_assistant_turn_container.markdown(user_ack_from_model1)
                     thinking_placeholder.empty()


            except Exception as e:
                thinking_placeholder.empty()
                error_msg = f"An unexpected error occurred: {e}"
                current_assistant_turn_container.error(error_msg)
                accumulated_final_parts_for_history.append({"type": "text", "data": f"Sorry, I encountered an error: {e}"})
                import traceback; traceback.print_exc()

        if accumulated_final_parts_for_history:
            # Prevent adding an empty assistant message if only an ack was processed and already stored
            # or if only errors occurred that were displayed directly.
            # Only add if there are actual new parts to log.
            if not (len(accumulated_final_parts_for_history) == 1 and \
                    accumulated_final_parts_for_history[0]["type"] == "text" and \
                    initial_ack_displayed_in_turn and \
                    accumulated_final_parts_for_history[0]["data"] == user_ack_from_model1 and \
                    not (is_code_related and action_for_next and prompt_for_next) ): # ensure it's not just a simple chat ack
                st.session_state.messages.append({"role": "assistant", "content_parts": accumulated_final_parts_for_history})
