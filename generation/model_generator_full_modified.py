import json
import argparse
from tqdm import tqdm
from openai import OpenAI
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def init_client(model_name, backend):
    if backend == "openai":
        if model_name == "gpt4o":
            return OpenAI(
                api_key="open-ai-api-key", # replace with your OpenAI API key
            ), "gpt-4o"
        elif model_name == "deepseek-chat":
            return OpenAI(
                api_key="deepseek-api-key", # replace with your DeepSeek API key
            ), "deepseek-chat"
        else:
            raise ValueError("Invalid model. Choose from gpt4o or deepseek-chat")
    elif backend == "hf":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        return (tokenizer, model), model_name
    else:
        raise ValueError("Invalid backend. Choose from openai or hf")


def generate_with_hf(model_pack, prompt):
    tokenizer, model = model_pack
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()

# retrieve highlighted tokens from index
def mark_tokens(text, highlight_indices_str):
    highlight_indices = sorted(set(int(i.strip()) for i in highlight_indices_str.split(",") if i.strip().isdigit()))
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    for i in highlight_indices:
        if 0 <= i < len(tokens):
            tokens[i] = f"**{tokens[i]}**"
    return " ".join(tokens)

# description for taxonomy
taxonomy_descriptions = {
    1: "Coreference Resolution – The explanation resolves references (e.g., pronouns or demonstratives) across premise and hypothesis.",
    2: "Semantic-level Inference – Based on word meaning (e.g., synonyms, antonyms, negation).",
    3: "Syntactic-level Inference – Based on structural rephrasing with the same meaning (e.g., syntactic alternation, coordination, subordination). If the explanation itself is the rephrase of the premise or hypothesis, it should be included in this category.",
    4: "Pragmatic-level Inference – This category would capture inferences that arise from logical implications embedded in the structure or semantics of the text itself, without relying on external context or background knowledge.",
    5: "Absence of Mention – Lack of supporting evidence, the hypothesis introduce information that is not supported, not entailed, or not mentioned in the premise, but could be true.",
    6: "Logical Structure Conflict – Structural logical exclusivity (e.g., either-or, at most, only, must), quantifier conflict, temporal conflict, location conflict, gender conflict etc.",
    7: "Factual Knowledge – Explanation relies on commonsense, background, or domain-specific facts. No further reasoning involved.",
    8: "World-Informed Logical Reasoning – Requires real-world causal, probabilistic reasoning or unstated but assumed information."
}

# few-shot examples for taxonomy
few_shot_examples = {
    1: {
        "premise": "The man in the black t-shirt is trying to throw something.",
        "hypothesis": "The man is in a black shirt.",
        "gold_label": "entailment",
        "explanation": "The man is in a black shirt refers to the man in the black t-shirt."
    },
    2: {
        "premise": "A man in a black tank top is wearing a red plaid hat.",
        "hypothesis": "A man in a hat.",
        "gold_label": "entailment",
        "explanation": "A red plaid hat is a specific type of hat."
    },
    3: {
        "premise": "Two women walk down a sidewalk along a busy street in a downtown area.",
        "hypothesis": "The women were walking downtown.",
        "gold_label": "entailment",
        "explanation": "The women were walking downtown is a rephrase of, Two women walk down a sidewalk along a busy street in a downtown area."
    },
    4: {
        "premise": "A girl in a blue dress takes off her shoes and eats blue cotton candy.",
        "hypothesis": "The girl is eating while barefoot.",
        "gold_label": "entailment",
        "explanation": "If a girl takes off her shoes, then she becomes barefoot, and if she eats blue candy, then she is eating."
    },
    5: {
        "premise": "A person with a purple shirt is painting an image of a woman on a white wall.",
        "hypothesis": "A woman paints a portrait of a person.",
        "gold_label": "neutral",
        "explanation": "A person with a purple shirt could be either a man or a woman. We can't assume the gender of the painter."
    },
    6: {
        "premise": "Five girls and two guys are crossing an overpass.",
        "hypothesis": "The three men sit and talk about their lives.",
        "gold_label": "contradiction",
        "explanation": "Three is not two."
    },
    7: {
        "premise": "Two people crossing by each other while kite surfing.",
        "hypothesis": "The people are both males.",
        "gold_label": "neutral",
        "explanation": "Not all people are males."
    },
    8: {
        "premise": "A girl in a blue dress takes off her shoes and eats blue cotton candy.",
        "hypothesis": "The girl in a blue dress is a flower girl at a wedding.",
        "gold_label": "neutral",
        "explanation": "A girl in a blue dress doesn’t imply the girl is a flower girl at a wedding."
    }
}

def build_prompt(mode, premise, hypothesis, gold_label, taxonomy_idx=None, highlighted_1="", highlighted_2=""):
    if mode == "highlight_index":
        return f"""You are an expert in Natural Language Inference (NLI). Your task is to generate possible explanations for why the following statement is **{gold_label}**, focusing on the highlighted parts of the sentences.\n\n    Context: {premise}\n    Highlighted word indices in Context: {highlighted_1}\n\n    Statement: {hypothesis}\n    Highlighted word indices in Statement: {highlighted_2}\n\n    Please list all possible explanations without introductory phrases.\n    Answer:"""
    
    elif mode == "highlight_marked":
        marked_premise = mark_tokens(premise, highlighted_1)
        marked_hypothesis = mark_tokens(hypothesis, highlighted_2)
        return f"""You are an expert in Natural Language Inference (NLI). Your task is to generate possible explanations for why the following statement is **{gold_label}**, focusing on the highlighted parts of the sentences. Highlighted parts are marked in \"**\".\n\n    Context: {marked_premise}\n    Statement: {marked_hypothesis}\n\n    Please list all possible explanations without introductory phrases.\n    Answer:"""
    
    elif mode == "label":
        return f"""You are an expert in Natural Language Inference (NLI). Please list all possible explanations for why the following statement is {gold_label} given the content below without introductory phrases.\n    Context: {premise}\n    Statement: {hypothesis}\n    Answer:"""
    
    elif mode == "taxonomy":
        # generate descriptions per category (1/8)
        description = taxonomy_descriptions[taxonomy_idx]
        few_shot = few_shot_examples[taxonomy_idx]
        few_shot_text = f"""Here is an example:\n
        Premise: {few_shot['premise']}
        Hypothesis: {few_shot['hypothesis']}
        Label: {few_shot['gold_label']}
        Explanation: {few_shot['explanation']}\n"""

        return f"""You are an expert in Natural Language Inference (NLI). Given the following taxonomy with description and one example, generate as many possible explanations as you can that specifically match the reasoning type described below. The explanation is for why the following statement is **{gold_label}**, given the content.
        ---
        The explanation category for generation is: {taxonomy_idx}: {description}
        {few_shot_text}
        ---
        Now, consider the following premise and hypothesis:
        Context: {premise}    
        Statement: {hypothesis}
        Please list all possible explanations for the given category without introductory phrases.
        Answer:"""
    
    elif mode == "classify":
        examples_text = ""
        for idx in range(1, 9):
            few_shot = few_shot_examples[idx]
            description = taxonomy_descriptions[idx]
            examples_text += f"""{idx}. {description}
        Example:
        Premise: {few_shot['premise']}
        Hypothesis: {few_shot['hypothesis']}
        Label: {few_shot['gold_label']}
        Explanation: {few_shot['explanation']}\n\n"""

        return f"""You are an expert in Natural Language Inference (NLI). Your task is to identify all applicable reasoning categories for explanations from the list below that could reasonably support the label. Please choose at least one category and multiple categories may apply.
        One example for each category is listed as below:
        ---
        {examples_text}
        ---
        Given the following premise and hypothesis, identify the applicable explanation categories:
        Premise: {premise}
        Hypothesis: {hypothesis}
        Label: {gold_label}
        Respond only with the numbers corresponding to the applicable categories, separated by commas, and no additional explanation.
        Answer:"""    
    
    elif mode == "classify_and_generate":
        return f"""You are an expert in Natural Language Inference (NLI). Your task is to examine the relationship between the following content and statement under the given gold label, and:
        First, identify all categories for explanations from the list below (you may choose more than one) that could reasonably support the label.
        Second, for each selected category, generate all possible explanations that reflect that type.
        
        The explanation categories are:
        1. Coreference Resolution – The explanation resolves references (e.g., pronouns or demonstratives) across premise and hypothesis.
        2. Semantic-level Inference – Based on word meaning (e.g., synonyms, antonyms, negation).
        3. Syntactic-level Inference – Based on structural rephrasing with the same meaning (e.g., syntactic alternation, coordination, subordination). If the explanation itself is the rephrase of the premise or hypothesis, it should be included in this category.
        4. Pragmatic-level Inference – This category would capture inferences that arise from logical implications embedded in the structure or semantics of the text itself, without relying on external context or background knowledge.
        5. Absence of Mention – Lack of supporting evidence, the hypothesis introduce information that is not supported, not entailed, or not mentioned in the premise, but could be true.
        6. Logical Structure Conflict – Structural logical exclusivity (e.g., either-or, at most, only, must), quantifier conflict, temporal conflict, location conflict, gender conflict etc.
        7. Factual Knowledge – Explanation relies on commonsense, background, or domain-specific facts. No further reasoning involved.
        8. World-Informed Logical Reasoning – Requires real-world causal, probabilistic reasoning or unstated but assumed information.

        Context: {premise}   
        Statement: {hypothesis}  
        Label: {gold_label}

        Please list all possible explanations without introductory phrases for all the chosen categories. Start directly with the category number and explanation, following the strict format below:
        1. Coreference Resolution:  
        - [Your explanation(s) here]  

        2. Semantic-level Inference:  
        - [Your explanation(s) here]  

        ... (continue for all reasonable categories)
        Answer:"""
        
    elif mode == "generate_highlight":
        return f"""You are an expert in NLI. Based on the label '{gold_label}', highlight relevant word indices in the premise and hypothesis.
        Highlighting rules:
        - For entailment: highlight at least one word in the premise.
        - For contradiction: highlight at least one word in both the premise and the hypothesis.
        - For neutral: highlight only in the hypothesis.
        Premise: {premise}
        Hypothesis: {hypothesis}
        Label: {gold_label}
        Please list **3** possible highlights using word index in the sentence without introductory phrases. Answer using word indices **starting from 0** and include punctuation marks as tokens (count them).
        Respond strictly this format:
        
        Highlight 1:
        Premise_Highlighted: [Your chosen index(es) here] 
        Hypothesis_Highlighted: [Your chosen index(es) here]
        Highlight 2:
        ...
        Answer:"""
        
    else:
        raise ValueError("Invalid mode. Choose from highlight_index, highlight_marked, label, taxonomy, classify, classify_and_generate and generate_highlight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt4o", "deepseek-chat"], required=False, default="gpt4o")
    parser.add_argument("--hf_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--backend", type=str, choices=["openai", "hf"], default="openai")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="api_generated_output.jsonl")
    parser.add_argument("--mode", type=str, choices=["highlight_index", "highlight_marked", "label", "taxonomy", "classify", "classify_and_generate", "generate_highlight"], required=True)
    args = parser.parse_args()

    client, model_name = init_client(args.hf_model if args.backend == "hf" else args.model, args.backend)

    with open(args.input, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    seen_pair_ids = set()
    try:
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                seen_pair_ids.add(record["pairID"])
    except FileNotFoundError:
        pass
    
    # generate explanations based on unique pairID (premise + hypothesis)
    if args.mode in ["label", "classify_and_generate", "classify", "generate_highlight"]:
        unique_data = {}
        for item in all_data:
            pid = item["pairID"]
            if pid not in unique_data:
                unique_data[pid] = {
                    "pairID": pid,
                    "premise": item["premise"].strip(),
                    "hypothesis": item["hypothesis"].strip(),
                    "gold_label": item["gold_label"].strip()
                }
        data_to_process = list(unique_data.values())
    
    # generate or classify based on unique pairID + 8 taxonomy 
    elif args.mode in ["taxonomy"]:
        unique_data = {}
        for item in all_data:
            pid = item["pairID"]
            if pid not in unique_data:
                unique_data[pid] = {
                    "pairID": pid,
                    "premise": item["premise"].strip(),
                    "hypothesis": item["hypothesis"].strip(),
                    "gold_label": item["gold_label"].strip()
                }
        unique_data = list(unique_data.values())
        expanded_data = []
        for item in unique_data:
            for idx in range(1, 9):
                expanded_data.append({
                    "pairID": item["pairID"],
                    "premise": item["premise"].strip(),
                    "hypothesis": item["hypothesis"].strip(),
                    "gold_label": item["gold_label"].strip(),
                    "taxonomy_idx": idx
                })
        data_to_process = expanded_data       
    else:
        data_to_process = all_data

    with open(args.output, "a", encoding="utf-8") as out_f:
        for item in tqdm(data_to_process, desc=f"Generating with mode: {args.mode}", total=len(data_to_process)):
            pid = item["pairID"]
            if pid in seen_pair_ids:
                continue
            try:
                prompt = build_prompt(
                    args.mode,
                    premise=item["premise"],
                    hypothesis=item["hypothesis"],
                    gold_label=item["gold_label"],
                    taxonomy_idx=item.get("taxonomy_idx"),
                    highlighted_1=item.get("Sentence1_Highlighted", ""),
                    highlighted_2=item.get("Sentence2_Highlighted", "")
                )
                #print(f"------Prompt:------ {prompt}")
                if args.backend == "openai":
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    )
                    answer = response.choices[0].message.content.strip()
                elif args.backend == "hf":
                    answer = generate_with_hf(client, prompt)

            except Exception as e:
                answer = f"Error: {str(e)}"

            result = {
                "pairID": pid,
                # "Sentence1_Highlighted": item.get("Sentence1_Highlighted", ""),
                # "Sentence2_Highlighted": item.get("Sentence2_Highlighted", ""),
                "Answer": answer
            }
            
            # include taxonomy index in results for mode "taxonomy"
            if args.mode in ["taxonomy"]:
                result["taxonomy_idx"] = item["taxonomy_idx"]
                
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
