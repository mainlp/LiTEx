import streamlit as st
import pandas as pd

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.set_page_config(page_title="Highlight Annotation Tool", layout="wide")
st.title("🔍 NLI Highlight Annotation Tool")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file and "df" not in st.session_state:
    st.session_state.df = load_data(uploaded_file)
    st.session_state.sample_index = 0

if "df" in st.session_state:
    
    df = st.session_state.df
    index = st.session_state.sample_index
    current = df.iloc[index]
    
    def is_annotated(row):
        return (
            isinstance(row.get("new_highlight1"), str) and row["new_highlight1"].strip() != "" and
            isinstance(row.get("new_highlight2"), str) and row["new_highlight2"].strip() != ""
        )

    total = len(df)
    completed = df.apply(is_annotated, axis=1).sum()
    progress = completed / total

    st.markdown("### 📝 Annotation Progress")
    st.progress(progress)
    st.markdown(f"**{completed} / {total} samples annotated**")

    if is_annotated(current):
        st.markdown("✅ This sample is **annotated**.")
    else:
        st.markdown("❌ This sample is **not yet annotated**.")

    st.markdown(f"**Pair ID:** `{current['pairID']}`")
    st.markdown(f"**Premise:** `{current['premise']}`")
    st.markdown(f"**Hypothesis:** `{current['hypothesis']}`")
    st.markdown(f"**Gold Label:** `{current['gold_label']}`")
    st.markdown(f"**Explanation:** {current['explanation']}")

    # Tokenization
    premise_tokens = current["premise"].split()
    hypothesis_tokens = current["hypothesis"].split()

    premise_with_idx = " ".join([f"{i}:{tok}" for i, tok in enumerate(premise_tokens)])
    hypothesis_with_idx = " ".join([f"{i}:{tok}" for i, tok in enumerate(hypothesis_tokens)])

    st.markdown("#### Premise with token indices")
    st.markdown(f"<div style='color:blue'>{premise_with_idx}</div>", unsafe_allow_html=True)

    st.markdown("#### Hypothesis with token indices")
    st.markdown(f"<div style='color:green'>{hypothesis_with_idx}</div>", unsafe_allow_html=True)

    premise_options = [f"{i}:{tok}" for i, tok in enumerate(premise_tokens)]
    hypothesis_options = [f"{i}:{tok}" for i, tok in enumerate(hypothesis_tokens)]

    def parse_prev(field, tokens):
        if pd.isna(field) or field == "":
            return []
        return [f"{i}:{tokens[i]}" for i in map(int, str(field).split(",")) if i < len(tokens)]

    default1 = parse_prev(current.get("new_highlight1", ""), premise_tokens)
    default2 = parse_prev(current.get("new_highlight2", ""), hypothesis_tokens)

    highlight1_display = st.multiselect("Select highlighted tokens in Premise:", options=premise_options, default=default1)
    highlight2_display = st.multiselect("Select highlighted tokens in Hypothesis:", options=hypothesis_options, default=default2)

    highlight1 = [int(item.split(":")[0]) for item in highlight1_display]
    highlight2 = [int(item.split(":")[0]) for item in highlight2_display]

    if st.button("✅ Save Annotation"):
        st.session_state.df.at[index, "new_highlight1"] = ",".join(map(str, highlight1))
        st.session_state.df.at[index, "new_highlight2"] = ",".join(map(str, highlight2))
        st.success("Saved!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Previous") and st.session_state.sample_index > 0:
            st.session_state.sample_index -= 1
    with col2:
        if st.button("➡️ Next") and st.session_state.sample_index < len(df) - 1:
            st.session_state.sample_index += 1

    st.markdown("---")
    st.markdown("### 💾 Download Updated CSV")
    st.download_button(
        label="Download CSV",
        data=st.session_state.df.to_csv(index=False).encode("utf-8"),
        file_name="annotated_highlight.csv"
    )
