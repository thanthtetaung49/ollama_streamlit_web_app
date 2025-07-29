import streamlit as st
import requests

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    selected_model = st.selectbox("Choose model", ['phi', 'phi-v2'])

    user_messages = [(i, m) for i, m in enumerate(st.session_state.messages) if m["role"] == "user"]

    if user_messages:
        for idx, msg in user_messages:
            preview = msg["content"][:40] + "..." if len(msg["content"]) > 40 else msg["content"]
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"{preview}")

            with col2:
                st.markdown(
                    f"""
                    <form action="" method="post">
                        <button name="delete_{idx}" type="submit" 
                                style="
                                    background: none;
                                    border: none;
                                    color: #888;
                                    font-size: 12px;
                                    padding: 0;
                                    margin-top: 4px;
                                    cursor: pointer;
                                " 
                                title="Delete">
                            ❌
                        </button>
                    </form>
                    """,
                    unsafe_allow_html=True
                )

                # Handle click (Streamlit workaround: use session_state as flag)
                if f"delete_{idx}" in st.session_state:
                    del st.session_state.messages[idx]
                    st.rerun(scope="app")

        if st.button("🗑️ Clear All", help="Delete all chat history"):
            st.session_state.messages = []
            st.rerun(scope="app")
    else:
        st.write("No user questions yet.")

st.header(f"Ollama {selected_model} Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input(f"Ask something to {selected_model}")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Generating the results..."):
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": selected_model,
                    "prompt": prompt,
                    "stream": False 
                }
            )
            result = response.json()
            bot_response = result.get("response", "No response received.")

        message_placeholder.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
