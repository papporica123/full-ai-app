import streamlit as st

from groq_client import generate_response
from search import format_results, web_search

RESEARCH_SYSTEM = """You are a careful research assistant.
Use any search results provided to ground your answer in current, factual information.
Cite sources by their [n] number when you rely on them, note any uncertainty or
conflicting information, and clearly separate facts from your own analysis."""

st.set_page_config(page_title="Research Assistant", page_icon="🔎")
st.title("🔎 Research Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    use_search = st.checkbox("Search the web before answering", value=True)
    if st.button("Clear conversation"):
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a research question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    context = ""
    if use_search:
        with st.spinner("Searching..."):
            try:
                results = web_search(question)
                context = format_results(results)
            except Exception as e:
                context = f"(Web search failed: {e})"

    llm_messages = [{"role": "system", "content": RESEARCH_SYSTEM}]
    if context:
        llm_messages.append({"role": "system", "content": f"Search results:\n{context}"})
    llm_messages.extend(
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    )

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = generate_response(llm_messages)
        except Exception as e:
            answer = f"Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
