import os

import modal
import streamlit as st
from openai import OpenAI


MAX_CHAT_HISTORY = 30
# personas.py or at top of app.py
PERSONAS = {
    "Helpful assistant": "You are a diligent, patient and friendly software developer who answers in a calm, encouraging tone and provides clear, detailed explanations rather than one-sentence replies.",
    "Sarcastic dev": "You are a lazy, sarcastic, angry software developer who answers with one sentence.",
    "Professor": "You are a patient CS professor who explains things step by step.",
}


@st.cache_resource
def init_client_and_model():
    """
    Create an OpenAI-compatible client pointing to your Modal vLLM deployment
    and fetch the active model ID (first model in /v1/models).
    """
    # In production, set MODAL_TOKEN in env or Streamlit secrets
    token = "my-token"

    # You can override these via env vars if needed
    workspace = os.getenv("MODAL_WORKSPACE","dim-lambros")
    app_name = os.getenv("MODAL_APP_NAME", "vllm-chat-rag-ragllmconfig")
    function_name = os.getenv("MODAL_FUNCTION_NAME", "serve")
    #print("workspace:", workspace)
    #print("app_name:", app_name)
    #print("function_name:", function_name)
    base_url = f"https://{workspace}--{app_name}-{function_name}.modal.run/v1"
    #print("base_url:", base_url)
    #st.sidebar.write("workspace:", workspace)
    #st.sidebar.write("app_name:", app_name)
    #st.sidebar.write("function_name:", function_name)
    #st.sidebar.write("base_url:", base_url)
    client = OpenAI(api_key=token, base_url=base_url)

    st.sidebar.write("Connecting to Modal backend...")
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No models returned from backend /v1/models")
    model_id = models.data[0].id
    return client, model_id


def get_completion_stream(client: OpenAI, model_id: str, messages: list):
    """
    Stream a chat completion from your Modal vLLM backend.
    """
    try:
        return client.chat.completions.create(
            model=model_id,
            stream=True,
            messages=messages,
        )
    except Exception as e:
        st.error(f"Call to LLM failed: {e}")
        return None


def main():
    st.set_page_config(page_title="Modal vLLM Chat", page_icon="💬")
    st.title("Modal vLLM Chat (Streamlit UI)")
    # choose persona
    persona_name = st.sidebar.selectbox("Personality", list(PERSONAS.keys()))
    system_prompt = PERSONAS[persona_name]


    # Initialize client + model
    client, model_id = init_client_and_model()
    st.sidebar.success(f"Connected model: `{model_id}`")

    # --- Chat state setup ---
    if "messages" not in st.session_state:
        # Initialize with current persona
        st.session_state.messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
    else:
        # Always keep system message in sync with selected persona
        st.session_state.messages[0]["content"] = system_prompt

    # Optional: sidebar controls
    if st.sidebar.button("Clear chat"):
        # Start a fresh chat with the CURRENT persona
        st.session_state.messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

    # --- Display chat history (skip system message) ---
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input ---
    user_input = st.chat_input("Type your message")

    if user_input:
        # Trim chat history (keep system + last N messages)
        if len(st.session_state.messages) > MAX_CHAT_HISTORY:
            sys_msg = st.session_state.messages[0]
            st.session_state.messages = [sys_msg] + st.session_state.messages[-MAX_CHAT_HISTORY + 1 :]

        # Add user message to state and UI
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant response (streaming)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            assistant_message = ""

            stream = get_completion_stream(client, model_id, st.session_state.messages)
            if stream is None:
                return

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    assistant_message += delta
                    placeholder.markdown(assistant_message)

        # Save assistant message to history
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_message}
        )


if __name__ == "__main__":
    main()
