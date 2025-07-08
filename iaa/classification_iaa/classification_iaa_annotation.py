import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="NLI Explanation Annotator", layout="wide")
st.title("🧠 NLI Explanation Reasoning Annotation")

# Sidebar: Upload & Sample Selection
st.sidebar.title("📂 Load Examples")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type = ["csv"])
data = None
sample_index = 0

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {len(data)} examples.")
    
    if "sample_index" not in st.session_state:
        st.session_state.sample_index = 0
        
    st.sidebar.markdown(f"**Annotation Progress**: {st.session_state.sample_index + 1} / {len(data)}")
    st.sidebar.progress((st.session_state.sample_index + 1) / len(data))
    
    if "sample_index" not in st.session_state:
        st.session_state.sample_index = 0
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("⬅️ Previous"):
        st.session_state.sample_index = max(0, st.session_state.sample_index - 1)
    if col2.button("Next ➡️"):
        st.session_state.sample_index = min(len(data) - 1, st.session_state.sample_index + 1)
    
    sample_index = st.session_state.sample_index
    row = data.iloc[sample_index]
    premise = row.get("premise", "")
    hypothesis = row.get("hypothesis", "")
    gold_label = row.get("gold_label", "")
    explanation = row.get("explanation", "")
else:
    # fallback default
    premise = st.text_area("Premise (Example)", "John bought a Ferrari.")
    hypothesis = st.text_area("Hypothesis (Example)", "John is wealthy.")
    gold_label = st.text_area("Gold Label (Example)", "Entailment")
    explanation = st.text_area("Explanation (Example)", "Because Ferraris are expensive, people who buy them are usually rich.")

if uploaded_file is not None:
    st.subheader("📝 Current Sample")
    premise = st.text_area("Premise", premise)
    hypothesis = st.text_area("Hypothesis", hypothesis)
    gold_label = st.text_area("Gold Label", gold_label)
    explanation = st.text_area("Explanation", explanation)

st.divider()

# Reasoning Steps
st.subheader("🧩 Step-wise Reasoning Assessment")

reasoning_steps = {
    "Step 1: Does the explanation rely on resolving coreference between entities?": {
        "tag": "Coreference Resolution",
        "desc": """
        **What to check:**
        Determine whether the main entities in the premise and hypothesis refer to the same real-world referent, including via pronouns or phrases.
        """
        },
    "Step 2: Does the explanation involve semantic similarity or substitution of key concepts?": {
        "tag": "Paraphrastic Inference (Semantic)",
        "desc": """
        **What to check:**
        Evaluate whether core words or expressions - including verbs, nouns, adjectives - are semantically related between the premise and hypothesis. This includes synonymy, antonymy, lexical entailment, or category membership.
        """
        },
    "Step 3: Does the explanation involve a change in sentence structure that preserves meaning?": {
        "tag": "Paraphrastic Inference (Syntactic)",
        "desc": """
        **What to check:**
        Determine whether the premise and hypothesis differ in structure - such as active vs. passive, reordered arguments, or coordination/subordination - while preserving the same meaning.
        """
        },
    "Step 4: Does the explanation rely on pragmatic cues like implicature or presupposition?":{
        "tag": "Pragmatic-Level Inference",
        "desc": """
        **What to check:**
        Look for meaning beyond the literal text - including implicature, presupposition, speaker intention, and conventional conversational meaning.
        """
        },
    "Step 5: Does the explanation point out information not mentioned in the premise?":{
        "tag": "Absence of Mention",
        "desc": """
        **What to check:**
        Check whether the hypothesis introduced information that is neither supported nor contradicted by the premise - i.e., it is not mentioned explicitly.
        """},
    "Step 6: Does the explanation refer to logical constraints or conflict?": {
        "tag": "Logical Conflict",
        "desc": """
        **What to check:**
        Evaluate whether the hypothesis interacts with the premise via logical structures — such as exclusivity, quantifiers (“only”, “none”), or conditionals — which constrain or conflict with each other.
        """},
    "Step 7: Does the explanation rely on commonsense, factual, or domain-specific knowledge?":{
        "tag": "Factual Knowledge",
        "desc": """
        **What to check:**
        Determine whether the hypothesis requires factual, commonsense, or domain-specific knowledge not provided in the premise.
        """
        },
    "Step 8: Does the explanation rely on real-world logical or causal reasoning?": {
        "tag": "World-Informed Logical Reasoning",
        "desc": """
        **What to check:**
        Assess whether the hypothesis is supported by **real-world logical or causal knowledge** that is not explicitly stated in the premise. This includes everyday commonsense, general knowledge, or domain-specific facts.
        """
    }
}

selected_steps = []
for step, info in reasoning_steps.items():
    with st.expander(f"{step} ({info['tag']})", expanded=False):
        st.markdown(info["desc"])
        if st.checkbox("Include this step", key=step):
            selected_steps.append({"step": step, "type": info["tag"]})

st.divider()

# Auto-assign explanation category based on the last selected step
st.subheader("🏷️ Explanation Classification")
st.markdown("_Note: If multiple reasoning steps are selected, the system will suggest an initial explanation category based on the **last selected step**. You can manually override this suggestion below._")

step_to_category_mapping = {
    "Coreference Resolution": "Coreference Resolution",
    "Paraphrastic Inference (Semantic)": "Semantic-level Inference",
    "Paraphrastic Inference (Syntactic)": "Syntactic-level Inference",
    "Pragmatic-Level Inference": "Pragmatic-Level Inference",
    "Absence of Mention": "Absence of Mention",
    "Logical Conflict": "Logical Structure Conflict",
    "Factual Knowledge": "Factual Knowledge",
    "World-Informed Logical Reasoning": "World-Informed Logical Reasoning"
}

guessed_category = None
if selected_steps:
    last_tag = selected_steps[-1]["type"]
    guessed_category = step_to_category_mapping.get(last_tag)


explanation_categories = ["-"] + sorted(set(step_to_category_mapping.values()))
default_index = explanation_categories.index(guessed_category) if guessed_category in explanation_categories else explanation_categories.index("-")

selected_category = st.selectbox(
    "Choose the explanation type",
    explanation_categories,
    index=default_index
)

# Optional Info
annotator_note = st.text_area("Note(optional)")

# Save Annotation Results
if st.button("✅ Save Annotation"):
    
    # determine high-level category
    text_based_types = [
        "Coreference Resolution",
        "Semantic-level Inference",
        "Syntactic-level Inference",
        "Pragmatic-Level Inference",
        "Absence of Mention",
        "Logical Structure Conflict"
    ]
    if selected_category in text_based_types:
        high_level_category = "Text-Based Inference"
    else:
        high_level_category = "World-Knowledge-Based Inference"
        
    current = {
        "pairID": row.get("pairID") if uploaded_file is not None else "",
        "premise": premise,
        "hypothesis": hypothesis,
        "explanation": explanation,
        "selected_steps": selected_steps,
        "explanation_category": selected_category,
        "note": annotator_note
    }
    
    existing_records = []
    try:
        with open("annotations.jsonl", "r", encoding="utf-8") as f:
            existing_records = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        pass

    # check for existing annotation
    updated = False
    for i, rec in enumerate(existing_records):
        if rec.get("pairID") == current["pairID"] and rec.get("explanation") == current["explanation"]:
            existing_records[i] = current
            updated = True
            break

    if not updated:
        existing_records.append(current)
    with open("annotations.jsonl", "w", encoding="utf-8") as f:
        for rec in existing_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    # Show annotation count summary
    try:
        with open("annotations.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
            count = len(lines)
            st.info(f"📊 Total annotations saved: {count}")
    except FileNotFoundError:
        st.info("📊 No saved annotations yet.")