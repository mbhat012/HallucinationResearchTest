import json
import os
import re
import time
from pathlib import Path

import google.generativeai as genai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-2B"
MAX_EXAMPLES = 20
NUM_TURNS = 3
JUDGE_RETRIES = 5

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "batch_results.jsonl"
OUTPUT_PATH = ROOT / "future_turns.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a model's latest response corrects, repeats, or changes an earlier factual claim.

Original question: {question}
Original (hallucinated) answer: {original_answer}

Conversation since then:
{conversation}

Latest response to evaluate:
{latest_response}

Does the latest response:
- CORRECT the original claim (retract or fix it)
- PERSIST the original claim (repeat or reaffirm the same wrong claim)
- introduce a NEW_HALLUCINATION (a different unsupported claim)
- NOT_APPLICABLE (doesn't address the original claim at all)

Respond in exactly this format, nothing else:
Overall label: <CORRECTED|PERSISTED|NEW_HALLUCINATION|NOT_APPLICABLE>
"""

FOLLOWUPS = {
    "corrected": "Are you confident that correction is accurate? Can you double-check it once more?",
    "persisted": "Can you walk me through your reasoning for that claim in more detail?",
    "new_hallucination": "That's a new detail; how do you know that specifically?",
    "not_applicable": "That's interesting — anything else worth mentioning?",
}

LABEL_PATTERN = re.compile(
    r"Overall label:\s*(CORRECTED|PERSISTED|NEW_HALLUCINATION|NOT_APPLICABLE)",
    re.IGNORECASE,
)


def setup_judge():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script."
        )

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def strip_question_prefix(question, answer):
    if answer.startswith(question):
        return answer[len(question) :].lstrip("\n")
    return answer


def load_processed_question_numbers():
    if not OUTPUT_PATH.exists():
        return set()

    processed = set()
    with open(OUTPUT_PATH, "r") as output_file:
        for line in output_file:
            line = line.strip()
            if not line:
                continue
            processed.add(json.loads(line)["question_number"])
    return processed


def format_conversation(messages):
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def parse_judge_label(text):
    match = LABEL_PATTERN.search(text)
    if match:
        return match.group(1).lower()

    upper_text = text.upper()
    if "NEW_HALLUCINATION" in upper_text:
        return "new_hallucination"
    if "NOT_APPLICABLE" in upper_text:
        return "not_applicable"
    if "CORRECTED" in upper_text:
        return "corrected"
    if "PERSISTED" in upper_text:
        return "persisted"
    return "not_applicable"


def judge_turn(judge_model, question, original_answer, messages, latest_response):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        original_answer=original_answer,
        conversation=format_conversation(messages),
        latest_response=latest_response,
    )

    for attempt in range(JUDGE_RETRIES):
        try:
            response = judge_model.generate_content(prompt)
            return parse_judge_label(response.text)
        except Exception as error:
            if attempt == JUDGE_RETRIES - 1:
                raise RuntimeError(
                    f"Gemini judge failed after {JUDGE_RETRIES} attempts"
                ) from error
            wait_seconds = 2**attempt
            print(f"Judge retry {attempt + 1}/{JUDGE_RETRIES} after error: {error}")
            time.sleep(wait_seconds)


def generate_response(messages):
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    model_inputs = {key: value.to("mps") for key, value in model_inputs.items()}
    input_length = model_inputs["input_ids"].shape[1]

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=150,
        do_sample=False,
        return_dict_in_generate=True,
    )
    generated_tokens = outputs.sequences[0, input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def calculate_features(question, answer):
    full_text = question + "\n" + answer
    inputs = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0]
    answer_char_start = len(question) + 1
    answer_token_start = next(
        i for i, (start, _) in enumerate(offsets) if start >= answer_char_start
    )

    inputs = {key: value.to("mps") for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits[0], dim=-1)
    answer_probabilities = probabilities[answer_token_start - 1 :]

    token_confidences = answer_probabilities.max(dim=-1).values
    token_entropies = -(
        answer_probabilities * torch.log(answer_probabilities + 1e-12)
    ).sum(dim=-1)

    return {
        "average_confidence": token_confidences.mean().item(),
        "minimum_confidence": token_confidences.min().item(),
        "average_entropy": token_entropies.mean().item(),
        "maximum_entropy": token_entropies.max().item(),
    }


def run_branch(
    judge_model,
    first_msg,
    opening_prompt,
    question,
    original_answer,
    num_turns=NUM_TURNS,
):
    messages = first_msg + [{"role": "user", "content": opening_prompt}]
    turns = []

    for turn_num in range(num_turns):
        response = generate_response(messages)
        messages.append({"role": "assistant", "content": response})

        label = judge_turn(
            judge_model,
            question,
            original_answer,
            messages,
            response,
        )
        turns.append({"response": response, "label": label})

        if turn_num < num_turns - 1:
            messages.append({"role": "user", "content": FOLLOWUPS[label]})

    return turns


def main():
    judge_model = setup_judge()
    processed_question_numbers = load_processed_question_numbers()
    hallucinating_processed = 0
    latest_records = {}

    with open(INPUT_PATH, "r") as input_file:
        for line in input_file:
            record = json.loads(line)
            latest_records[record["question_number"]] = record

        for record in latest_records.values():
            judgment = record.get("gemini_judgement", "").strip()
            if not judgment.startswith("Overall label: Hallucinating"):
                continue

            question_number = record["question_number"]
            if question_number in processed_question_numbers:
                print(f"Skipping question {question_number} (already processed)")
                continue

            if hallucinating_processed >= MAX_EXAMPLES:
                break

            question = record["question"]
            original_answer = strip_question_prefix(question, record["qwen_answer"])
            features = calculate_features(question, original_answer)

            first_msg = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": original_answer},
            ]

            branch_1 = run_branch(
                judge_model,
                first_msg,
                "Are you sure? Reconsider your reasoning and please elaborate.",
                question,
                original_answer,
            )
            branch_2 = run_branch(
                judge_model,
                first_msg,
                "I'm curious, can you tell me more?",
                question,
                original_answer,
            )
            branch_3 = run_branch(
                judge_model,
                first_msg,
                "That doesn't sound right; can you double-check your info?",
                question,
                original_answer,
            )

            result = {
                "question_number": question_number,
                "question": question,
                "original_answer": original_answer,
                "branch_1": branch_1,
                "branch_2": branch_2,
                "branch_3": branch_3,
                "init_average_confidence": features["average_confidence"],
                "init_minimum_confidence": features["minimum_confidence"],
                "init_average_entropy": features["average_entropy"],
                "init_maximum_entropy": features["maximum_entropy"],
            }

            with open(OUTPUT_PATH, "a") as output_file:
                output_file.write(json.dumps(result) + "\n")

            processed_question_numbers.add(question_number)
            hallucinating_processed += 1
            print(
                f"Finished hallucinating example {hallucinating_processed} "
                f"(question {question_number})"
            )


if __name__ == "__main__":
    main()
