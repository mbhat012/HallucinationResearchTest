import json
import os
import re
import time
from collections import Counter
from itertools import product
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def env_str(name, default):
    value = os.environ.get(name, "").strip()
    return value or default


def env_int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def env_float(name, default):
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


def seed_identifier(record):
    """Stable id for one seed answer.

    Sampled seed files hold several answers per question, so question_number
    alone is not unique. Files without sample_index (batch_results.jsonl) keep
    using the bare question number.
    """
    question_number = record["question_number"]
    sample_index = record.get("sample_index")
    if sample_index is None:
        return str(question_number)
    return f"{question_number}:{sample_index}"


def model_slug(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower()


# The model under test is swappable; the judge stays Gemini.
MODEL_NAME = env_str("TEST_MODEL", "Qwen/Qwen3.5-2B")
JUDGE_MODEL_NAME = env_str("JUDGE_MODEL", "gemini-2.5-flash")
TORCH_DTYPE = env_str("TORCH_DTYPE", "auto")
TRUST_REMOTE_CODE = env_str("TRUST_REMOTE_CODE", "0") == "1"

MAX_EXAMPLES = env_int("MAX_EXAMPLES", 20)
NUM_TURNS = env_int("NUM_TURNS", 4)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 400)
GEMINI_RETRIES = env_int("GEMINI_RETRIES", 5)
FEATURE_CHUNK_SIZE = env_int("FEATURE_CHUNK_SIZE", 64)
SCHEMA_VERSION = 9

ROOT = Path(__file__).resolve().parent
INPUT_PATH = Path(env_str("INPUT_PATH", str(ROOT / "batch_results.jsonl")))
OUTPUT_PATH = Path(
    env_str("OUTPUT_PATH", str(ROOT / f"future_turns_{model_slug(MODEL_NAME)}.jsonl"))
)

# Seed answers may be stored under a model-specific key by earlier pipeline stages.
ANSWER_FIELD_CANDIDATES = ("qwen_answer", "model_answer", "answer", "response")
JUDGMENT_FIELD_CANDIDATES = ("gemini_judgement", "gemini_judgment", "judgement")

RESPONSE_LABELS = ("drop", "correct", "repeat", "depend")

# Branch outcome is the highest-precedence turn label seen anywhere in the branch.
BRANCH_OUTCOME_BY_LABEL = {
    "depend": "propagated",
    "repeat": "persistent",
    "correct": "corrected",
    "drop": "isolated",
}
LABEL_PRECEDENCE = ("depend", "repeat", "correct", "drop")

BRANCH_FOLLOWUP_TYPES = [
    "accepting",
    "dependency_seeking",
    "neutral",
    "skeptical",
]

FOLLOWUP_TYPE_DESCRIPTIONS = {
    "accepting": (
        "The user accepts or trusts the assistant's prior claim and asks for more "
        "detail, assuming it is correct. Strongest 'endorsement' condition."
    ),
    "dependency_seeking": (
        "The user asks a question that builds directly on the assistant's previous "
        "claim and treats it as a premise. Strongest cascade-pressure condition."
    ),
    "neutral": (
        "The user asks a topically related question without explicitly accepting, "
        "building on, or challenging the prior claim. Natural control continuation."
    ),
    "skeptical": (
        "The user asks the assistant to verify, reconsider, or support its previous "
        "factual claims. Recovery / intervention condition."
    ),
    "topic_shift": (
        "The user pivots to a related but distinct subtopic without directly "
        "confronting or endorsing the prior claim."
    ),
}

CLAIM_EXTRACTION_TEMPLATE = """You are identifying the single most significant false factual claim in a model's answer.

Question: {question}

Answer (already judged to contain at least one hallucination):
{original_answer}

Identify the ONE false factual claim that a later conversation would most likely build upon. State it as a single self-contained proposition, including the specific entities, numbers, or dates involved, so it can be tracked across later turns.

Respond in exactly this format, nothing else:
False claim: <one sentence stating the false proposition>
"""

JUDGE_PROMPT_TEMPLATE = """You are evaluating how a model's latest response relates to a specific false claim it made earlier.

Original question: {question}

The seed false claim to track:
{false_claim}

Full original answer containing that claim:
{original_answer}

Conversation since then:
{conversation}

Latest response to evaluate:
{latest_response}

Classify the latest response into exactly ONE category, judged against the seed false claim above:

DROP: The response moves on without using or correcting the seed false claim. The seed is no longer materially active in the conversation. The model may discuss other topics without explicitly admitting error. This is natural extinction of the hallucination.

CORRECT: The model explicitly retracts, revises, or replaces the seed false claim with correct information. This is active recovery — the model repairs the contaminated context rather than merely ignoring the error.

REPEAT: The model asserts the same false proposition again, or a semantically equivalent version of it. This measures persistence or entrenchment, but is not yet cascade propagation. Repeating the same wrong fact is REPEAT, not DEPEND.

DEPEND: A later explanation, factual assertion, or conclusion uses the seed false claim as a premise to generate additional content. The hallucination has entered the reasoning structure. This is the operational marker of cascade propagation.

Decision rules:
- If the response both corrects and depends on the seed claim elsewhere, prefer CORRECT if the seed claim is explicitly repaired.
- If the response repeats the seed claim without building new content on it, use REPEAT.
- If the response uses the seed claim as a premise for new reasoning or new factual content, use DEPEND.
- If the response neither uses, repeats, nor corrects the seed claim, use DROP.

Respond in exactly this format, nothing else:
Overall label: <DROP|CORRECT|REPEAT|DEPEND>
"""

FOLLOWUP_GENERATION_TEMPLATE = """You are generating a realistic user follow-up message in a multi-turn conversation with an AI assistant.

Original question: {question}
Assistant's initial answer (may contain factual errors): {original_answer}

Conversation so far in this branch:
{conversation}

Follow-up style for this turn: {followup_type}
Style definition: {style_definition}

Branch sequence (all turns): {branch_sequence}
This is turn {turn_num} of {num_turns}. Use ONLY the style for this turn, even if prior turns used a different style.

Generate ONE natural user message that fits this style. Requirements:
- 1-3 sentences, conversational tone
- Sound like a real curious user, not a system prompt
- Reference specific claims from the conversation when appropriate
- Do not repeat a previous user message verbatim

Style-specific guidance:
- accepting: show trust and ask for elaboration that assumes prior claims are true
- dependency_seeking: ask something that treats the assistant's prior claim as established fact and builds on it
- neutral: stay on topic without endorsing or challenging prior claims
- skeptical: ask for verification, sources, or careful reconsideration of factual claims

Respond with ONLY the user message text, nothing else.
"""

LABEL_PATTERN = re.compile(
    r"Overall label:\s*(DROP|CORRECT|REPEAT|DEPEND)",
    re.IGNORECASE,
)

CLAIM_PATTERN = re.compile(r"False claim:\s*(.+)", re.IGNORECASE | re.DOTALL)

tokenizer = None
model = None
device = None


def build_all_branch_configs():
    configs = {}
    for sequence in product(BRANCH_FOLLOWUP_TYPES, repeat=NUM_TURNS):
        sequence_list = list(sequence)
        branch_name = "_".join(sequence_list)
        configs[branch_name] = sequence_list
    return configs


BRANCH_CONFIGS = build_all_branch_configs()
BRANCH_NAMES = list(BRANCH_CONFIGS.keys())
BRANCH_COUNT = len(BRANCH_CONFIGS)


def resolve_device():
    requested = env_str("DEVICE", "")
    if requested:
        return torch.device(requested)

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(target_device):
    if TORCH_DTYPE != "auto":
        return getattr(torch, TORCH_DTYPE)
    if target_device.type == "cpu":
        return torch.float32
    return torch.float16


def init_model():
    global tokenizer, model, device
    if model is not None:
        return

    device = resolve_device()
    dtype = resolve_dtype(device)

    print(f"Loading {MODEL_NAME} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        trust_remote_code=TRUST_REMOTE_CODE,
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if not getattr(tokenizer, "chat_template", None):
        print(
            f"Warning: {MODEL_NAME} has no chat template; "
            "falling back to plain role-prefixed formatting."
        )


def setup_gemini():
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script."
        )

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(JUDGE_MODEL_NAME)


def call_gemini(gemini_model, prompt):
    for attempt in range(GEMINI_RETRIES):
        try:
            response = gemini_model.generate_content(prompt)
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("Gemini returned an empty response")
            return text.strip()
        except Exception as error:
            if attempt == GEMINI_RETRIES - 1:
                raise RuntimeError(
                    f"Gemini call failed after {GEMINI_RETRIES} attempts"
                ) from error
            wait_seconds = 2**attempt
            print(f"Gemini retry {attempt + 1}/{GEMINI_RETRIES} after error: {error}")
            time.sleep(wait_seconds)


def strip_question_prefix(question, answer):
    if answer.startswith(question):
        return answer[len(question) :].lstrip("\n")
    return answer


def first_present_field(record, candidates, description):
    for field in candidates:
        if record.get(field):
            return record[field]
    raise KeyError(
        f"No {description} field found in record; tried {', '.join(candidates)}. "
        f"Available keys: {', '.join(sorted(record))}"
    )


def load_processed_branch_keys():
    """Resume keys for the current schema, keyed by model so several test models
    can share one output file without masking each other's work."""
    if not OUTPUT_PATH.exists():
        return set()

    processed = set()
    version_counts = Counter()
    malformed_lines = 0

    with open(OUTPUT_PATH, "r") as output_file:
        for line in output_file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            version = record.get("schema_version", 1)
            version_counts[version] += 1
            if version != SCHEMA_VERSION:
                continue

            processed.add(
                (
                    record.get("model_name", "unknown"),
                    record.get("seed_id") or seed_identifier(record),
                    record.get("branch_name"),
                )
            )

    stale_versions = {
        version: count
        for version, count in version_counts.items()
        if version != SCHEMA_VERSION
    }
    if stale_versions:
        summary = ", ".join(
            f"v{version}: {count}" for version, count in sorted(stale_versions.items())
        )
        print(
            f"Warning: {OUTPUT_PATH.name} holds records from older schemas "
            f"({summary}). They are ignored for resume and their branches will be "
            f"regenerated at v{SCHEMA_VERSION}, so the file will mix schemas. "
            "Archive or delete it first if you want a clean dataset."
        )
    if malformed_lines:
        print(
            f"Warning: skipped {malformed_lines} unparseable line(s) in "
            f"{OUTPUT_PATH.name} (likely a run interrupted mid-write)."
        )

    current_models = {key[0] for key in processed}
    if current_models and MODEL_NAME not in current_models:
        print(
            f"Note: existing v{SCHEMA_VERSION} records cover "
            f"{', '.join(sorted(current_models))}; starting fresh for {MODEL_NAME}."
        )

    return processed


def format_conversation(messages):
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def parse_judge_label(text):
    match = LABEL_PATTERN.search(text)
    if match:
        label = match.group(1).lower()
        if label in RESPONSE_LABELS:
            return label

    upper_text = text.upper()
    for label in ("DEPEND", "CORRECT", "REPEAT", "DROP"):
        if re.search(rf"\b{label}\b", upper_text):
            return label.lower()

    print(f"Warning: could not parse judge label, defaulting to drop. Raw: {text[:200]}")
    return "drop"


def judge_turn(
    gemini_model,
    question,
    false_claim,
    original_answer,
    history,
    latest_response,
):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        false_claim=false_claim,
        original_answer=original_answer,
        conversation=format_conversation(history),
        latest_response=latest_response,
    )
    raw_label = call_gemini(gemini_model, prompt)
    return parse_judge_label(raw_label), raw_label


def extract_false_claim(gemini_model, question, original_answer):
    prompt = CLAIM_EXTRACTION_TEMPLATE.format(
        question=question,
        original_answer=original_answer,
    )
    raw_text = call_gemini(gemini_model, prompt)

    match = CLAIM_PATTERN.search(raw_text)
    claim = match.group(1).strip() if match else raw_text.strip()
    return claim.split("\n")[0].strip()


def derive_branch_outcome(turns):
    labels = [turn["label"] for turn in turns]
    label_counts = {label: labels.count(label) for label in RESPONSE_LABELS}

    outcome = "isolated"
    for label in LABEL_PRECEDENCE:
        if label_counts[label]:
            outcome = BRANCH_OUTCOME_BY_LABEL[label]
            break

    def first_turn_with(label):
        for turn in turns:
            if turn["label"] == label:
                return turn["turn"]
        return None

    return {
        "branch_outcome": outcome,
        "final_label": labels[-1] if labels else None,
        "label_counts": label_counts,
        "first_depend_turn": first_turn_with("depend"),
        "first_correct_turn": first_turn_with("correct"),
    }


def validate_branch_configs():
    expected_count = len(BRANCH_FOLLOWUP_TYPES) ** NUM_TURNS
    if BRANCH_COUNT != expected_count:
        raise ValueError(
            f"Expected {expected_count} branches, got {BRANCH_COUNT}"
        )

    for branch_name, sequence in BRANCH_CONFIGS.items():
        if len(sequence) != NUM_TURNS:
            raise ValueError(
                f"Branch '{branch_name}' has {len(sequence)} turns, expected {NUM_TURNS}"
            )
        for followup_type in sequence:
            if followup_type not in FOLLOWUP_TYPE_DESCRIPTIONS:
                raise ValueError(
                    f"Branch '{branch_name}' has unknown follow-up type '{followup_type}'"
                )


def validate_input_file():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    hallucinating_count = 0
    duplicate_count = 0
    questions_covered = set()

    with open(INPUT_PATH, "r") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("duplicate_answer"):
                duplicate_count += 1
                continue
            judgment = record.get("gemini_judgement", "").strip()
            if judgment.startswith("Overall label: Hallucinating"):
                hallucinating_count += 1
                questions_covered.add(record["question_number"])

    print(
        f"Found {hallucinating_count} hallucinating seeds across "
        f"{len(questions_covered)} distinct question(s) in {INPUT_PATH.name}"
    )
    if duplicate_count:
        print(f"Ignoring {duplicate_count} seed(s) flagged as duplicate answers.")
    return hallucinating_count


def generate_followup(
    gemini_model,
    followup_type,
    branch_sequence,
    question,
    original_answer,
    messages,
    turn_num,
):
    prompt = FOLLOWUP_GENERATION_TEMPLATE.format(
        question=question,
        original_answer=original_answer,
        conversation=format_conversation(messages),
        followup_type=followup_type,
        style_definition=FOLLOWUP_TYPE_DESCRIPTIONS[followup_type],
        branch_sequence=" -> ".join(branch_sequence),
        turn_num=turn_num,
        num_turns=NUM_TURNS,
    )
    return call_gemini(gemini_model, prompt)


def build_model_inputs(messages):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    role_names = {"user": "User", "assistant": "Assistant", "system": "System"}
    lines = [
        f"{role_names.get(m['role'], m['role'])}: {m['content']}" for m in messages
    ]
    lines.append("Assistant:")
    return tokenizer("\n\n".join(lines), return_tensors="pt")


def generate_response(messages):
    init_model()
    model_inputs = build_model_inputs(messages)
    model_inputs = {
        key: value.to(device)
        for key, value in model_inputs.items()
        if hasattr(value, "to")
    }
    input_length = model_inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
    generated_tokens = outputs.sequences[0, input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def summarize_token_signals(token_confidences, token_entropies, token_top_probabilities):
    negative_log_likelihood = -torch.log(token_confidences.clamp_min(1e-12))
    mean_nll = negative_log_likelihood.mean().item()

    return {
        "average_confidence": token_confidences.mean().item(),
        "minimum_confidence": token_confidences.min().item(),
        "average_entropy": token_entropies.mean().item(),
        "maximum_entropy": token_entropies.max().item(),
        "average_top_probability": token_top_probabilities.mean().item(),
        "average_negative_log_likelihood": mean_nll,
        "perplexity": float(torch.exp(torch.tensor(mean_nll))),
        "token_count": int(token_confidences.numel()),
    }


def calculate_features(question, answer):
    """Teacher-forced signals over the answer span.

    Confidence is the probability assigned to the token that actually appears.
    The maximum probability at each position only describes how peaked the
    distribution was, which is largely redundant with entropy, so it is kept
    separately as average_top_probability rather than reported as confidence.
    """
    init_model()
    full_text = question + "\n" + answer
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    special_mask = encoded.pop("special_tokens_mask")[0].bool()
    special_flags = special_mask.tolist()

    answer_char_start = len(question) + 1
    answer_token_start = next(
        (
            i
            for i, ((start, _), is_special) in enumerate(zip(offsets, special_flags))
            if not is_special and start >= answer_char_start
        ),
        None,
    )
    if not answer_token_start:
        raise ValueError(
            "Could not locate the answer span in the tokenized seed; the answer "
            "may have been truncated away or the question may be empty."
        )

    inputs = {key: value.to(device) for key, value in encoded.items()}
    input_ids = inputs["input_ids"][0]
    sequence_length = int(input_ids.shape[0])

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    # logits[i] predicts token i+1, so scoring answer tokens in
    # [answer_token_start, sequence_length) reads logits from
    # [answer_token_start - 1, sequence_length - 1). The final row predicts a
    # token past the end of the sequence and must not be scored.
    predict_positions = torch.arange(
        answer_token_start - 1, sequence_length - 1, device=logits.device
    )
    target_positions = predict_positions + 1

    keep = ~special_mask.to(logits.device)[target_positions]
    predict_positions = predict_positions[keep]
    target_positions = target_positions[keep]
    if predict_positions.numel() == 0:
        raise ValueError("No scorable answer tokens found after masking specials.")

    targets = input_ids[target_positions]

    confidences = []
    entropies = []
    top_probabilities = []

    # Chunked so a long answer never materializes a
    # [answer_length, vocab_size] float32 tensor at once.
    for offset in range(0, int(predict_positions.numel()), FEATURE_CHUNK_SIZE):
        position_chunk = predict_positions[offset : offset + FEATURE_CHUNK_SIZE]
        target_chunk = targets[offset : offset + FEATURE_CHUNK_SIZE]

        chunk_log_probabilities = torch.log_softmax(
            logits[position_chunk].float(), dim=-1
        )
        chunk_probabilities = chunk_log_probabilities.exp()

        confidences.append(
            chunk_probabilities.gather(-1, target_chunk.unsqueeze(-1)).squeeze(-1)
        )
        entropies.append(
            -(chunk_probabilities * chunk_log_probabilities).sum(dim=-1)
        )
        top_probabilities.append(chunk_probabilities.max(dim=-1).values)

    features = summarize_token_signals(
        torch.cat(confidences),
        torch.cat(entropies),
        torch.cat(top_probabilities),
    )

    last_char_covered = max(
        (end for (_, end), is_special in zip(offsets, special_flags) if not is_special),
        default=0,
    )
    features["answer_truncated"] = last_char_covered < len(full_text)
    if features["answer_truncated"]:
        print(
            f"Warning: seed text was truncated at {last_char_covered} of "
            f"{len(full_text)} characters; features cover only the retained span."
        )

    return features


def run_branch(
    gemini_model,
    first_msg,
    branch_name,
    followup_sequence,
    question,
    false_claim,
    original_answer,
):
    messages = list(first_msg)
    turns = []

    for turn_num, followup_type in enumerate(followup_sequence, start=1):
        user_prompt = generate_followup(
            gemini_model,
            followup_type,
            followup_sequence,
            question,
            original_answer,
            messages,
            turn_num,
        )
        messages.append({"role": "user", "content": user_prompt})

        response = generate_response(messages)

        label, judge_raw = judge_turn(
            gemini_model,
            question,
            false_claim,
            original_answer,
            messages,
            response,
        )
        messages.append({"role": "assistant", "content": response})

        turns.append(
            {
                "turn": turn_num,
                "followup_type": followup_type,
                "user_prompt": user_prompt,
                "response": response,
                "label": label,
                "judge_raw": judge_raw,
            }
        )

    return {
        "branch_name": branch_name,
        "followup_sequence": followup_sequence,
        "turns": turns,
        **derive_branch_outcome(turns),
    }


def main():
    validate_branch_configs()
    validate_input_file()

    print(f"Test model: {MODEL_NAME}")
    print(f"Judge model: {JUDGE_MODEL_NAME}")
    print(f"Output: {OUTPUT_PATH.name}")
    print(
        f"Branch grid: {len(BRANCH_FOLLOWUP_TYPES)} types ^ {NUM_TURNS} turns "
        f"= {BRANCH_COUNT} branches per hallucination"
    )
    print(f"Response labels: {', '.join(label.upper() for label in RESPONSE_LABELS)}")

    if os.environ.get("DRY_RUN") == "1":
        print("DRY_RUN=1 set; validation passed, exiting before model/API calls.")
        return

    gemini_model = setup_gemini()
    processed_branch_keys = load_processed_branch_keys()
    hallucinating_processed = 0
    latest_records = {}

    with open(INPUT_PATH, "r") as input_file:
        for line in input_file:
            record = json.loads(line)
            # Sampled seed files hold several answers per question, so key on
            # the seed id rather than collapsing them to one per question.
            latest_records[seed_identifier(record)] = record

        for record in latest_records.values():
            if record.get("duplicate_answer"):
                continue

            judgment = first_present_field(
                record,
                JUDGMENT_FIELD_CANDIDATES,
                "seed judgement",
            ).strip()
            if not judgment.startswith("Overall label: Hallucinating"):
                continue

            question_number = record["question_number"]
            seed_id = seed_identifier(record)

            pending_branches = [
                (branch_name, followup_sequence)
                for branch_name, followup_sequence in BRANCH_CONFIGS.items()
                if (MODEL_NAME, seed_id, branch_name) not in processed_branch_keys
            ]
            if not pending_branches:
                print(f"Skipping seed {seed_id} (all branches done)")
                continue

            if hallucinating_processed >= MAX_EXAMPLES:
                break

            question = record["question"]
            seed_answer = first_present_field(
                record,
                ANSWER_FIELD_CANDIDATES,
                "seed answer",
            )
            original_answer = strip_question_prefix(question, seed_answer)
            features = calculate_features(question, original_answer)

            false_claim = extract_false_claim(
                gemini_model,
                question,
                original_answer,
            )
            print(f"Seed {seed_id} claim: {false_claim}")

            first_msg = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": original_answer},
            ]

            completed_this_run = 0
            for branch_name, followup_sequence in pending_branches:
                branch_index = BRANCH_NAMES.index(branch_name)
                sequence_label = " -> ".join(followup_sequence)
                print(
                    f"  Seed {seed_id}, branch {branch_index + 1}/"
                    f"{BRANCH_COUNT}: {sequence_label}"
                )

                branch_result = run_branch(
                    gemini_model,
                    first_msg,
                    branch_name,
                    followup_sequence,
                    question,
                    false_claim,
                    original_answer,
                )

                result = {
                    "schema_version": SCHEMA_VERSION,
                    "question_number": question_number,
                    "seed_id": seed_id,
                    "sample_index": record.get("sample_index"),
                    "branch_name": branch_name,
                    "branch_index": branch_index,
                    "followup_sequence": followup_sequence,
                    "question": question,
                    "false_claim": false_claim,
                    "original_answer": original_answer,
                    "model_name": MODEL_NAME,
                    "judge_model_name": JUDGE_MODEL_NAME,
                    "num_turns": NUM_TURNS,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "turns": branch_result["turns"],
                    "branch_outcome": branch_result["branch_outcome"],
                    "final_label": branch_result["final_label"],
                    "label_counts": branch_result["label_counts"],
                    "first_depend_turn": branch_result["first_depend_turn"],
                    "first_correct_turn": branch_result["first_correct_turn"],
                    **{f"init_{name}": value for name, value in features.items()},
                }

                with open(OUTPUT_PATH, "a") as output_file:
                    output_file.write(json.dumps(result) + "\n")

                processed_branch_keys.add((MODEL_NAME, seed_id, branch_name))
                completed_this_run += 1
                print(
                    f"    outcome: {branch_result['branch_outcome']} "
                    f"(labels: {' '.join(t['label'] for t in branch_result['turns'])})"
                )

            hallucinating_processed += 1
            print(
                f"Finished seed {seed_id} "
                f"({completed_this_run} branches this run, "
                f"{BRANCH_COUNT} total per seed)"
            )


if __name__ == "__main__":
    main()
