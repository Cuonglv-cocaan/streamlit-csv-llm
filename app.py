"""
Chat with Your CSV
------------------
Upload any CSV. Get an instant dashboard. Chat with the data. Browse
auto-generated visualizations. Pick your model and watch your token spend.

This is the main Streamlit entry point. It is deliberately thin: all the
data profiling and LLM logic lives in llm_utils.py so the UI stays easy
to read.

Run locally:
    streamlit run app.py

Before running, make sure you have an OpenAI API key stored in
.streamlit/secrets.toml (see .streamlit/secrets.toml.example for the
template).
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from llm_utils import (
    AVAILABLE_MODELS,
    MODEL_PRICING,
    ask_llm_stream,
    build_data_context,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Chat with Your CSV",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path(__file__).parent resolves to this script's directory, so the path
# works regardless of where Streamlit is launched from.
SAMPLE_CSV_PATH = Path(__file__).parent / "sample_data" / "sample.csv"
SAMPLE_LABEL = "sample.csv (NYC TLC yellow taxi August 2025)"

# Generic starter questions that work for any CSV. Avoid domain-specific phrasing.
STARTER_QUESTIONS = [
    "Describe this dataset in plain English.",
    "Are there any columns with missing values I should worry about?",
    "What patterns or anomalies stand out in the numeric columns?",
    "Which columns look most useful for analysis, and why?",
]


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    """Set defaults so we never have to test for key existence later."""
    # Streamlit reruns this whole script on every user interaction. setdefault
    # is idempotent, so any existing state survives the rerun untouched.
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("tokens_by_model", {})   # {model: {"input": N, "output": N}}
    st.session_state.setdefault("use_sample", False)


_init_state()


# ---------------------------------------------------------------------------
# Password gate (optional shared-password authentication)
# ---------------------------------------------------------------------------
#
# If APP_PASSWORD is set in secrets, users see a password screen before
# accessing the app. If APP_PASSWORD is NOT set, the app runs without
# authentication (the default for local development).
#
# This is shared-password auth: anyone who knows the password gets in. It's
# the right tool for "share this app with my team but not the public
# internet." For per-user accounts, look at the streamlit-authenticator
# library or Streamlit's native st.login OAuth flow.

def check_password() -> bool:
    """Render the password gate. Return True if the user is authenticated."""
    # If no APP_PASSWORD is configured, the app runs unprotected.
    try:
        expected = st.secrets.get("APP_PASSWORD")
    except Exception:
        expected = None
    if not expected:
        return True

    # Already authenticated this browser-tab session: let through.
    if st.session_state.get("authenticated"):
        return True

    # Render the gate.
    st.title("Sign in")
    st.caption("Enter the password to access this app.")
    pw = st.text_input("Password", type="password", label_visibility="collapsed")
    if pw:
        if pw == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# Gate the rest of the app behind the password check.
if not check_password():
    st.stop()


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading CSV...")
def load_uploaded_csv(file) -> pd.DataFrame:
    """Read an uploaded CSV. Cached so reruns don't re-parse the same file."""
    return pd.read_csv(file)


@st.cache_data(show_spinner="Loading sample data...")
def load_sample_csv() -> pd.DataFrame:
    """Read the bundled NYC taxi sample CSV from disk."""
    return pd.read_csv(SAMPLE_CSV_PATH)


@st.cache_data(show_spinner="Profiling your data for the LLM...")
def cached_data_context(file_label: str, n_rows: int, n_cols: int, df: pd.DataFrame) -> str:
    """Build the LLM context string once per file. The first three args are
    part of the cache key so a different file invalidates the cache."""
    # file_label, n_rows, n_cols are passed only so Streamlit's cache key
    # changes when a different file is loaded; they're not used inside.
    return build_data_context(df)


# ---------------------------------------------------------------------------
# Token / cost helpers
# ---------------------------------------------------------------------------

def record_token_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    bucket = st.session_state.tokens_by_model.setdefault(
        model, {"input": 0, "output": 0}
    )
    bucket["input"] += input_tokens
    bucket["output"] += output_tokens


def total_tokens_used() -> int:
    return sum(
        b["input"] + b["output"]
        for b in st.session_state.tokens_by_model.values()
    )


def total_cost_usd() -> float:
    total = 0.0
    for model, bucket in st.session_state.tokens_by_model.items():
        if model not in MODEL_PRICING:
            continue
        price = MODEL_PRICING[model]
        total += (bucket["input"] * price["input"] + bucket["output"] * price["output"]) / 1_000_000
    return total


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Load your data")
    uploaded_file = st.file_uploader(
        label="Drop a CSV file here",
        type="csv",
        help="Any CSV works. The app auto-profiles it for you.",
    )

    st.caption("…or try the sample dataset:")
    if st.button("Load NYC taxi sample", use_container_width=True):
        st.session_state.use_sample = True
        st.session_state.messages = []   # fresh chat for new data

    st.divider()

    st.header("2. Choose your model")
    selected_model = st.selectbox(
        label="OpenAI model",
        options=AVAILABLE_MODELS,
        index=0,
        help=(
            "gpt-4o-mini is cheap and fast (recommended for most use cases). "
            "gpt-4o is more capable but roughly 16x more expensive per token."
        ),
    )

    # Show this model's pricing as a tiny caption
    price = MODEL_PRICING[selected_model]
    st.caption(
        f"Pricing: \\${price['input']:.3f} / 1M input tokens, "
        f"\\${price['output']:.3f} / 1M output tokens."
    )

    st.divider()

    st.header("3. Usage so far")
    used = total_tokens_used()
    cost = total_cost_usd()
    col_a, col_b = st.columns(2)
    col_a.metric("Tokens", f"{used:,}")
    col_b.metric("Cost", f"${cost:.4f}")

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        # Force an immediate rerun so the empty chat is reflected on this click.
        st.rerun()

    # Sign out button: only shows when password protection is enabled.
    try:
        auth_enabled = bool(st.secrets.get("APP_PASSWORD"))
    except Exception:
        auth_enabled = False
    if auth_enabled:
        if st.button("Sign out", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()


# ---------------------------------------------------------------------------
# Decide which DataFrame we are working with
# ---------------------------------------------------------------------------

df: pd.DataFrame | None = None
file_label: str | None = None

if uploaded_file is not None:
    # Uploaded file always wins over the sample
    try:
        df = load_uploaded_csv(uploaded_file)
        file_label = uploaded_file.name
        st.session_state.use_sample = False
    except Exception as e:
        st.error(f"Could not read that CSV: {e}")
        st.stop()
elif st.session_state.use_sample:
    try:
        df = load_sample_csv()
        file_label = SAMPLE_LABEL
    except Exception as e:
        st.error(f"Could not load sample data: {e}")
        st.stop()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Chat with Your CSV")
st.caption(
    "Upload a CSV (or load the sample), then ask questions in plain English, "
    "browse the data overview, or explore auto-generated visualizations."
)

if df is None:
    st.info("Use the sidebar to upload a CSV or click **Load NYC taxi sample** to get started.")
    st.stop()


# --- Top metric cards (data-agnostic) -------------------------------------

total_cells = df.shape[0] * df.shape[1]
missing_cells = int(df.isna().sum().sum())
missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0
numeric_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

st.caption(f"Data source: `{file_label}`")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns)}")
m3.metric("Numeric columns", f"{len(numeric_cols)}")
m4.metric("Missing values", f"{missing_pct:.1f}%")


# --- Build LLM context once per file --------------------------------------

data_context = cached_data_context(
    file_label=file_label,
    n_rows=len(df),
    n_cols=len(df.columns),
    df=df,
)


# ---------------------------------------------------------------------------
# Tabs: Chat / Data Overview / Visualize
# ---------------------------------------------------------------------------

tab_chat, tab_overview, tab_viz = st.tabs(
    ["💬 Chat", "📋 Data overview", "📊 Visualize"]
)


# === Chat tab =============================================================

with tab_chat:
    # Suggestion chips relay their text through session state: a clicked
    # button can't directly trigger the chat input, so we stash the chip's
    # question in `pending_prompt`, rerun, and consume it on the next pass
    # as if the user had typed it themselves.
    pending = st.session_state.pop("pending_prompt", None)

    # Hide chips when the chat is non-empty OR a chip click is mid-relay,
    # otherwise they'd briefly re-render during the consume-rerun cycle.
    if not st.session_state.messages and not pending:
        st.markdown("**Not sure where to start? Try a question:**")
        chip_cols = st.columns(2)
        for i, question in enumerate(STARTER_QUESTIONS):
            if chip_cols[i % 2].button(
                question,
                key=f"starter_{i}",
                use_container_width=True,
            ):
                # Stash the chip's text and rerun. The pop above picks it up
                # on the next pass and feeds it into the normal chat flow.
                st.session_state.pending_prompt = question
                st.rerun()

    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # `prompt` comes from one of two sources this run: typed text wins,
    # a clicked-chip's relayed value is the fallback.
    typed = st.chat_input("Ask anything about your data...")
    prompt = typed or pending

    if prompt:
        # Show user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream assistant's reply
        with st.chat_message("assistant"):
            usage_tracker: dict = {}
            stream = ask_llm_stream(
                user_message=prompt,
                data_context=data_context,
                # Slice [:-1] excludes the user message we just appended, so
                # the LLM doesn't see it twice (once as history, once as current).
                history=st.session_state.messages[:-1],
                model=selected_model,
                usage_tracker=usage_tracker,
            )
            # st.write_stream renders chunks live AND returns the full string
            # once the stream finishes, so we can save it to chat history.
            reply = st.write_stream(stream)

            if usage_tracker.get("input_tokens") is not None:
                record_token_usage(
                    model=selected_model,
                    input_tokens=usage_tracker["input_tokens"],
                    output_tokens=usage_tracker["output_tokens"],
                )

        st.session_state.messages.append({"role": "assistant", "content": reply})
        # Rerun so the sidebar token / cost metrics update immediately
        st.rerun()


# === Data overview tab =====================================================

with tab_overview:
    st.subheader("Sample rows")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Column profile")
    profile = pd.DataFrame(
        {
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Non-null": [int(df[c].notna().sum()) for c in df.columns],
            "Null": [int(df[c].isna().sum()) for c in df.columns],
            "Unique": [int(df[c].nunique()) for c in df.columns],
        }
    )
    st.dataframe(profile, use_container_width=True, hide_index=True)

    if numeric_cols:
        st.subheader("Numeric column statistics")
        st.dataframe(
            df[numeric_cols].describe().round(2),
            use_container_width=True,
        )
    else:
        st.info("This dataset has no numeric columns.")


# === Visualize tab ========================================================

with tab_viz:
    st.caption(
        "Auto-generated visualizations. Pick which columns to plot. "
        "Works for any CSV, regardless of subject matter."
    )

    if numeric_cols:
        st.subheader("Numeric column distributions")
        selected_numeric = st.multiselect(
            "Numeric columns to plot",
            options=numeric_cols,
            default=numeric_cols[: min(4, len(numeric_cols))],
            key="viz_numeric_select",
        )
        if selected_numeric:
            chart_cols = st.columns(2)
            for i, col_name in enumerate(selected_numeric):
                with chart_cols[i % 2]:
                    fig = px.histogram(df, x=col_name, nbins=30, title=col_name)
                    fig.update_layout(
                        showlegend=False,
                        height=300,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one numeric column to plot.")

    if cat_cols:
        st.subheader("Categorical column counts")
        selected_cat = st.multiselect(
            "Categorical columns to plot",
            options=cat_cols,
            default=cat_cols[: min(2, len(cat_cols))],
            key="viz_cat_select",
        )
        for col_name in selected_cat:
            counts = (
                df[col_name]
                .value_counts(dropna=False)
                .head(10)
                .reset_index()
            )
            counts.columns = [col_name, "count"]
            fig = px.bar(counts, x=col_name, y="count", title=f"Top values: {col_name}")
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    if not numeric_cols and not cat_cols:
        st.info("This dataset has no columns to visualize automatically.")
