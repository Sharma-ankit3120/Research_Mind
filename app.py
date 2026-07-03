"""
ResearchMind — Multi-Agent AI Research Assistant
--------------------------------------------------
A Streamlit front-end for a 4-stage multi-agent research pipeline:
Search -> Read -> Write -> Review, followed by a Q&A chat over the
final report and critic review.

Run with:
    streamlit run app.py
"""

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from agents import (
    build_search_agent,
    build_read_agent,
    writer_chain,
    critic_chain,
    llm,
)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Theme (dark navy / cyan / purple, matching the reference mockup)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --bg-1: #0b0f24;
            --bg-2: #140f2e;
            --panel: rgba(24, 22, 51, 0.55);
            --panel-border: rgba(139, 124, 246, 0.18);
            --cyan: #2dd4e8;
            --violet: #8b7cf6;
            --pink: #ec4899;
            --amber: #f2b84b;
            --text-dim: #8b8fb8;
            --text: #edeefb;
        }

        .stApp {
            background: radial-gradient(circle at 12% 8%, rgba(139, 124, 246, 0.16), transparent 40%),
                        radial-gradient(circle at 88% 12%, rgba(45, 212, 232, 0.12), transparent 42%),
                        radial-gradient(circle at 50% 100%, rgba(236, 72, 153, 0.08), transparent 45%),
                        linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
            color: var(--text);
        }

        #MainMenu, footer, header {visibility: hidden;}

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* ---------- Header ---------- */
        .rm-logo {
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(120deg, var(--cyan) 10%, var(--violet) 55%, var(--pink) 95%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        div[data-testid="stTextInput"] input {
            background-color: rgba(20, 18, 46, 0.75) !important;
            border: 1px solid var(--panel-border) !important;
            color: var(--text) !important;
            border-radius: 9px !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: var(--violet) !important;
            box-shadow: 0 0 0 2px rgba(139, 124, 246, 0.35) !important;
        }

        div.stButton > button {
            background: linear-gradient(120deg, var(--cyan), var(--violet));
            color: #0b0f24;
            font-weight: 700;
            border: none;
            border-radius: 9px;
            padding: 0.45rem 1.6rem;
            box-shadow: 0 4px 16px rgba(139, 124, 246, 0.35);
            transition: filter 0.15s ease;
        }
        div.stButton > button:hover {
            filter: brightness(1.12);
            color: #0b0f24;
        }

        /* ---------- Stepper ---------- */
        .stepper-wrap {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 1.6rem 0 2rem 0;
        }
        .step-circle {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1rem;
            border: 2px solid var(--panel-border);
            color: var(--text-dim);
            background: rgba(24, 22, 51, 0.6);
        }
        .step-circle.active {
            border-color: transparent;
            color: #0b0f24;
            background: linear-gradient(135deg, var(--cyan), var(--violet));
            box-shadow: 0 0 18px rgba(139, 124, 246, 0.55);
        }
        .step-circle.done {
            border-color: var(--cyan);
            color: var(--cyan);
            background: rgba(45, 212, 232, 0.08);
        }
        .step-label {
            font-size: 0.78rem;
            color: var(--text-dim);
            text-align: center;
            margin-top: 6px;
        }
        .step-label.active {
            color: var(--violet);
            font-weight: 600;
        }
        .step-line {
            flex: 1;
            height: 2px;
            background: var(--panel-border);
            margin: 0 4px;
            align-self: flex-start;
            margin-top: 21px;
        }
        .step-line.done { background: linear-gradient(90deg, var(--cyan), var(--violet)); }
        .step-col {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        /* ---------- Panels ---------- */
        .rm-panel {
            background: linear-gradient(160deg, rgba(28, 25, 58, 0.65), rgba(18, 16, 38, 0.65));
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            height: 100%;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(6px);
        }
        .rm-panel-title {
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.6px;
            margin-bottom: 0.8rem;
        }
        .rm-panel-title.cyan {
            background: linear-gradient(120deg, var(--cyan), var(--violet));
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .rm-panel-title.purple {
            background: linear-gradient(120deg, var(--pink), var(--violet));
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .rm-placeholder {
            color: var(--text-dim);
            font-size: 0.88rem;
            line-height: 1.5;
            border-left: 2px solid var(--panel-border);
            padding-left: 10px;
        }
        .rm-content {
            color: var(--text);
            font-size: 0.88rem;
            line-height: 1.55;
            max-height: 420px;
            overflow-y: auto;
            border-left: 2px solid var(--cyan);
            padding-left: 10px;
            white-space: pre-wrap;
        }

        /* ---------- Chat ---------- */
        .chat-bubble-user {
            background: rgba(139, 124, 246, 0.14);
            border: 1px solid rgba(139, 124, 246, 0.3);
            border-radius: 10px;
            padding: 0.6rem 0.9rem;
            margin: 0.35rem 0;
            font-size: 0.88rem;
            max-width: 80%;
            margin-left: auto;
        }
        .chat-bubble-assistant {
            background: rgba(45, 212, 232, 0.06);
            border-left: 2px solid var(--cyan);
            border-radius: 0 8px 8px 0;
            padding: 0.5rem 0.9rem;
            margin: 0.35rem 0;
            font-size: 0.88rem;
            max-width: 85%;
        }
        .chat-bubble-system {
            border-left: 2px solid var(--pink);
            padding: 0.4rem 0.9rem;
            margin: 0.35rem 0;
            font-size: 0.85rem;
            color: var(--text-dim);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
defaults = {
    "topic": "",
    "step": 0,            # 0 = not started, 1..4 = current/completed step
    "running": False,
    "search_results": None,
    "scraped_content": None,
    "report": None,
    "feedback": None,
    "chat_history": [],   # list[HumanMessage | AIMessage]
    "chat_display": [],   # list[("user"/"assistant"/"system", text)]
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

STEP_LABELS = ["Search", "Read", "Write", "Review"]

CHAT_SYSTEM_PROMPT = """
You are an AI Research Assistant.

You have already completed research on a topic.

You are given:
- Search Results
- Scraped Articles
- Final Report
- Critic Review

Your job is to answer the user's questions.

Rules:
1. First use the provided research context.
2. If the answer exists in the context, answer using it.
3. If some information is missing, use your own knowledge to complete the explanation.
4. Never hallucinate facts.
5. If something is uncertain, say it is uncertain.
6. Explain concepts clearly.
7. Maintain conversation history.

Research Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CHAT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
chat_chain = chat_prompt | llm


# ----------------------------------------------------------------------
# Pipeline step functions (adapted from run_research_pipeline)
# ----------------------------------------------------------------------
def run_search(topic: str) -> str:
    search_agent = build_search_agent()
    result = search_agent.invoke(
        {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
    )
    return result["messages"][-1].content


def run_read(topic: str, search_results: str) -> str:
    reader_agent = build_read_agent()
    result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Based on the following search results about '{topic}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{search_results[:800]}",
                )
            ]
        }
    )
    return result["messages"][-1].content


def run_write(topic: str, search_results: str, scraped_content: str) -> str:
    research_combined = (
        f"Search Results:\n{search_results}\n\n"
        f"Detailed Scraped Content:\n{scraped_content}\n"
    )
    return writer_chain.invoke({"topic": topic, "research": research_combined})


def run_critic(report: str) -> str:
    return critic_chain.invoke({"report": report})


def build_research_context() -> str:
    return f"""
    TOPIC: {st.session_state.topic}

    ================ FINAL REPORT ==================
    {st.session_state.report}

    ================ REVIEW ========================
    {st.session_state.feedback}
    """


def execute_pipeline(topic: str):
    st.session_state.topic = topic
    st.session_state.running = True
    st.session_state.chat_display = [
        ("system", "Research pipeline initialized. Running Search \u2192 Read \u2192 Write \u2192 Review...")
    ]
    st.session_state.chat_history = []

    st.session_state.step = 1
    with st.spinner("Searching the web..."):
        st.session_state.search_results = run_search(topic)

    st.session_state.step = 2
    with st.spinner("Reading and scraping top sources..."):
        st.session_state.scraped_content = run_read(topic, st.session_state.search_results)

    st.session_state.step = 3
    with st.spinner("Writing the report..."):
        st.session_state.report = run_write(
            topic, st.session_state.search_results, st.session_state.scraped_content
        )

    st.session_state.step = 4
    with st.spinner("Critiquing the report..."):
        st.session_state.feedback = run_critic(st.session_state.report)

    st.session_state.running = False
    st.session_state.chat_display.append(
        ("system", "Report ready. Ask a question about the research below.")
    )


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
header_col1, header_col2, header_col3 = st.columns([1, 5, 1])
with header_col1:
    st.markdown('<div class="rm-logo">🧠 RM</div>', unsafe_allow_html=True)

with header_col2:
    topic_input = st.text_input(
        "topic", placeholder="Enter a research topic...", label_visibility="collapsed"
    )

with header_col3:
    go_clicked = st.button("Go", use_container_width=True)

if go_clicked and topic_input.strip():
    execute_pipeline(topic_input.strip())

# ----------------------------------------------------------------------
# Stepper
# ----------------------------------------------------------------------
current_step = st.session_state.step
cols = st.columns([1, 0.4, 1, 0.4, 1, 0.4, 1])
step_positions = [0, 2, 4, 6]

for i, label in enumerate(STEP_LABELS):
    step_num = i + 1
    with cols[step_positions[i]]:
        if step_num < current_step or (step_num == current_step and not st.session_state.running):
            css_class = "done" if step_num < current_step else "active"
        elif step_num == current_step and st.session_state.running:
            css_class = "active"
        else:
            css_class = ""
        label_class = "active" if css_class in ("active",) else ""
        st.markdown(
            f"""
            <div class="step-col">
                <div class="step-circle {css_class}">{step_num}</div>
                <div class="step-label {label_class}">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if i < 3:
        with cols[step_positions[i] + 1]:
            line_class = "done" if step_num < current_step else ""
            st.markdown(f'<div class="step-line {line_class}"></div>', unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------
# Report Draft / Critic Review panels
# ----------------------------------------------------------------------
panel_col1, panel_col2 = st.columns(2)

with panel_col1:
    st.markdown('<div class="rm-panel">', unsafe_allow_html=True)
    st.markdown('<div class="rm-panel-title cyan">REPORT DRAFT</div>', unsafe_allow_html=True)
    if st.session_state.report:
        st.markdown(f'<div class="rm-content">{st.session_state.report}</div>', unsafe_allow_html=True)
    elif st.session_state.step >= 3:
        st.markdown(
            '<div class="rm-placeholder">Generating comprehensive research report based on '
            'search results and scraped content. The AI is analyzing and synthesizing '
            'information to create a structured document...</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="rm-placeholder">Enter a topic and click Go to begin research.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with panel_col2:
    st.markdown('<div class="rm-panel">', unsafe_allow_html=True)
    st.markdown('<div class="rm-panel-title purple">CRITIC REVIEW</div>', unsafe_allow_html=True)
    if st.session_state.feedback:
        st.markdown(f'<div class="rm-content">{st.session_state.feedback}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="rm-placeholder">Awaiting report completion. Once the draft is ready, '
            'the critic chain will evaluate the report for accuracy, coherence, and completeness. '
            'Critical feedback will be displayed here...</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------
# Chat Q&A
# ----------------------------------------------------------------------
st.markdown('<div class="rm-panel">', unsafe_allow_html=True)
st.markdown('<div class="rm-panel-title cyan">CHAT Q&A</div>', unsafe_allow_html=True)

if not st.session_state.chat_display:
    st.markdown(
        '<div class="rm-placeholder">Research pipeline initialized. Enter a topic to begin analysis.</div>',
        unsafe_allow_html=True,
    )
else:
    for role, text in st.session_state.chat_display:
        if role == "user":
            st.markdown(f'<div class="chat-bubble-user">{text}</div>', unsafe_allow_html=True)
        elif role == "assistant":
            st.markdown(f'<div class="chat-bubble-assistant">{text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-system">{text}</div>', unsafe_allow_html=True)

chat_col1, chat_col2 = st.columns([5, 1])
with chat_col1:
    question = st.text_input(
        "question",
        placeholder="Ask a question about the research...",
        label_visibility="collapsed",
        disabled=not st.session_state.report,
        key="question_input",
    )
with chat_col2:
    send_clicked = st.button(
        "Send", use_container_width=True, disabled=not st.session_state.report
    )

if send_clicked and question.strip():
    st.session_state.chat_display.append(("user", question))
    try:
        response = chat_chain.invoke(
            {
                "context": build_research_context(),
                "chat_history": st.session_state.chat_history,
                "question": question,
            }
        )
        st.session_state.chat_display.append(("assistant", response.content))
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=response.content))
    except Exception as e:
        st.session_state.chat_display.append(("system", f"Error: {str(e)}"))
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)