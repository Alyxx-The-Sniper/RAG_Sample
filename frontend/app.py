import streamlit as st
import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"

BACKGROUND_IMAGE_URL = (
    "https://images.unsplash.com/photo-1695144244472-a4543101ef35?q=80&w=1332&auto=format&fit=crop"
)

st.markdown(
    f"""
<style>
html, body, .stApp {{
    height: 100%;
    background: url('{BACKGROUND_IMAGE_URL}') no-repeat center center fixed;
    background-size: cover;
    color: #eee;
    font-family: 'Cinzel', serif;
}}
.stSpinner > div > div {{
    color: #fff;
}}
.chat-bubble {{
    background: rgba(0, 0, 0, 0.7);
    padding: 0.75rem;
    border-radius: 8px;
    margin-top: 0.5rem;
    color: #eee;
    font-size: 0.95rem;
    line-height: 1.4;
    word-wrap: break-word;
}}
@media (max-width: 600px) {{
    .chat-bubble {{
        font-size: 0.85rem;
    }}
}}
.stTextInput>div>div>input {{
    font-size: 1rem;
}}
</style>
""",
    unsafe_allow_html=True
)

st.title("Hi I'm Alyx 🤖")

if "history" not in st.session_state:
    st.session_state.history = []

def markdown_and_urls_to_html(text):
    # Convert markdown links [text](url)
    text = re.sub(
        r'\[([^\]]+)\]\((https?://[^\)]+)\)',
        r'<a href="\2" target="_blank" style="color:#65b7ff;text-decoration:underline;">\1</a>',
        text
    )
    # Convert plain URLs to clickable links (avoiding already converted)
    text = re.sub(
        r'(?<!href=")(https?://[^\s<>"\'\)]+)',
        r'<a href="\1" target="_blank" style="color:#65b7ff;text-decoration:underline;">\1</a>',
        text
    )
    return text

with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Ask me anything about Alexis:", key="user_input")
    submitted = st.form_submit_button("send ➤")

if submitted and user_query:
    with st.spinner("Thinking..."):
        response = requests.post(
            CHAT_ENDPOINT,
            json={"query": user_query}
        )
        if response.ok:
            answer = response.json().get("response", "No answer provided.")
            st.session_state.history.append({"user": user_query, "bot": answer})
        else:
            st.error("Sorry, something went wrong. Please try again.")

# Display chat history with clickable links in bubbles
if st.session_state.history:
    st.markdown("---")
    for entry in reversed(st.session_state.history):  # newest on top
        bot_message_html = markdown_and_urls_to_html(entry['bot'])
        st.markdown(f"""
<div class="chat-bubble">
<b>🧛‍♀️ Alyx the bot vampire:</b><br>{bot_message_html}
</div>
<div style="height: 0.3rem;"></div>
<div class="chat-bubble" style="background: rgba(255, 255, 255, 0.2); color: #ddd;">
<b>🧑 You:</b><br>{entry['user']}
</div>
""", unsafe_allow_html=True)
