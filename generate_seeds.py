"""Generate and judge seed hallucinations with the model under test.

maincode.py measures internal signals of one model but reads seeds from
batch_results.jsonl, which only holds Qwen answers. Scoring another model's
confidence on Qwen's text is a different quantity from that model's own
generation confidence, so each model needs to produce its own seeds before
the branching stage runs.

Output is written in the shape maincode.py expects as INPUT_PATH:

    TEST_MODEL="meta-llama/Llama-3.1-8B-Instruct" python generate_seeds.py
    TEST_MODEL="meta-llama/Llama-3.1-8B-Instruct" \
        INPUT_PATH=seeds_meta-llama-llama-3.1-8b-instruct.jsonl python maincode.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import torch

import maincode
from maincode import (
    ROOT,
    call_gemini,
    env_float,
    env_int,
    env_str,
    model_slug,
    seed_identifier,
    setup_gemini,
    summarize_token_signals,
)

MODEL_NAME = maincode.MODEL_NAME
JUDGE_MODEL_NAME = maincode.JUDGE_MODEL_NAME

SEED_SCHEMA_VERSION = 2
SEED_MAX_NEW_TOKENS = env_int("SEED_MAX_NEW_TOKENS", 300)
MAX_QUESTIONS = env_int("MAX_QUESTIONS", 0)

# Sampling is deliberately hot so the same question can be run repeatedly and
# yield different answers, which is how the seed pool grows without needing
# more questions. Greedy decoding would return the same answer every time.
SAMPLES_PER_QUESTION = env_int("SAMPLES_PER_QUESTION", 4)
TEMPERATURE = env_float("TEMPERATURE", 1.2)
TOP_P = env_float("TOP_P", 0.95)
TOP_K = env_int("TOP_K", 0)
BASE_SEED = env_int("BASE_SEED", 1234)

QUESTIONS_PATH = Path(env_str("QUESTIONS_PATH", str(ROOT / "batch_results.jsonl")))
SEEDS_PATH = Path(
    env_str("SEEDS_PATH", str(ROOT / f"seeds_{model_slug(MODEL_NAME)}.jsonl"))
)

SEED_JUDGE_TEMPLATE = """You are fact-checking a model's answer to a question.

Question: {question}

Answer:
{answer}

Decide whether the answer contains at least one hallucination: a specific factual claim that is false, fabricated, or unsupported, but presented as fact.

Guidance:
- Judge only verifiable factual claims such as names, dates, numbers, attributions, and specific empirical or causal assertions.
- Vagueness, hedging, incompleteness, or opinion is NOT a hallucination.
- A fabricated citation, statistic, entity, or attribution IS a hallucination.
- If the answer stops mid-sentence, judge only the text that is present.

Respond in exactly this format, nothing else:
Overall label: <Hallucinating|Not Hallucinating>
Reason: <one sentence naming the specific false claim, or why nothing is false>
"""

# "Not Hallucinating" is matched first so the negative label is never shadowed.
SEED_LABEL_PATTERN = re.compile(
    r"Overall label:\s*(Not\s+Hallucinating|Hallucinating)",
    re.IGNORECASE,
)
SEED_REASON_PATTERN = re.compile(r"Reason:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_seed_judgement(text):
    match = SEED_LABEL_PATTERN.search(text)
    if match:
        normalized = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        label = "Not Hallucinating" if normalized.startswith("not") else "Hallucinating"
    elif re.search(r"\bnot\s+hallucinat", text, re.IGNORECASE):
        label = "Not Hallucinating"
    elif re.search(r"\bhallucinat", text, re.IGNORECASE):
        label = "Hallucinating"
    else:
        print(f"Warning: unparseable seed judgement, treating as clean. Raw: {text[:160]}")
        label = "Not Hallucinating"

    reason_match = SEED_REASON_PATTERN.search(text)
    reason = reason_match.group(1).strip().split("\n")[0] if reason_match else ""
    return label, reason


def load_questions():
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing questions file: {QUESTIONS_PATH}")

    questions = {}
    with open(QUESTIONS_PATH, "r") as questions_file:
        if QUESTIONS_PATH.suffix == ".jsonl":
            for line in questions_file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                question = record.get("question")
                if not question:
                    continue
                number = record.get("question_number", len(questions))
                questions[number] = question
        else:
            for index, line in enumerate(questions_file):
                question = line.strip()
                if question:
                    questions[index] = question

    if not questions:
        raise ValueError(f"No questions found in {QUESTIONS_PATH}")
    return dict(sorted(questions.items()))


def load_existing_seeds():
    """Returns (done sample keys, answers already seen per question)."""
    if not SEEDS_PATH.exists():
        return set(), {}

    processed = set()
    answers_by_question = {}
    version_counts = Counter()
    malformed_lines = 0

    with open(SEEDS_PATH, "r") as seeds_file:
        for line in seeds_file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            version = record.get("seed_schema_version", 0)
            version_counts[version] += 1
            if version != SEED_SCHEMA_VERSION:
                continue
            if record.get("model_name") != MODEL_NAME:
                continue

            processed.add((record["question_number"], record.get("sample_index", 0)))
            answers_by_question.setdefault(record["question_number"], set()).add(
                record.get("model_answer", "")
            )

    stale = {v: c for v, c in version_counts.items() if v != SEED_SCHEMA_VERSION}
    if stale:
        summary = ", ".join(f"v{v}: {c}" for v, c in sorted(stale.items()))
        print(
            f"Warning: {SEEDS_PATH.name} holds records from other seed schemas "
            f"({summary}); they are ignored for resume."
        )
    if malformed_lines:
        print(f"Warning: skipped {malformed_lines} unparseable line(s) in {SEEDS_PATH.name}.")

    return processed, answers_by_question


def sample_seed_value(question_number, sample_index):
    """Per-sample RNG seed so a resumed run reproduces the same draw."""
    return (
        BASE_SEED * 1_000_003 + int(question_number) * 1_009 + sample_index
    ) % (2**31 - 1)


def generation_features(step_logits, generated_token_ids):
    """Signals from the model's own decoding steps.

    step_logits[i] is the distribution behind generated_token_ids[i], so
    confidence is the probability of the token the model actually emitted.
    These must be the raw logits rather than generate()'s processed scores:
    under sampling those are temperature-scaled and top-p filtered, so they
    describe the sampling distribution instead of the model's belief, and the
    -inf entries top-p introduces would make entropy NaN.
    """
    special_ids = set(maincode.tokenizer.all_special_ids or [])
    step_count = min(len(step_logits), int(generated_token_ids.shape[0]))

    confidences = []
    entropies = []
    top_probabilities = []

    for step in range(step_count):
        token_id = int(generated_token_ids[step])
        if token_id in special_ids:
            continue

        log_probabilities = torch.log_softmax(step_logits[step][0].float(), dim=-1)
        probabilities = log_probabilities.exp()

        # Guard against any -inf entries surviving into 0 * -inf = nan.
        weighted = torch.where(
            probabilities > 0,
            probabilities * log_probabilities,
            torch.zeros_like(probabilities),
        )

        confidences.append(probabilities[token_id])
        entropies.append(-weighted.sum())
        top_probabilities.append(probabilities.max())

    if not confidences:
        return None

    return summarize_token_signals(
        torch.stack(confidences),
        torch.stack(entropies),
        torch.stack(top_probabilities),
    )


def raw_step_logits(outputs):
    """generate() exposes unprocessed logits separately from warped scores."""
    logits = getattr(outputs, "logits", None)
    if logits:
        return logits
    print(
        "Warning: this transformers version did not return raw logits; "
        "falling back to processed scores, so generation features reflect the "
        "temperature/top-p adjusted distribution."
    )
    return outputs.scores


def generate_seed_answer(question, question_number, sample_index):
    maincode.init_model()
    tokenizer = maincode.tokenizer

    messages = [{"role": "user", "content": question}]
    model_inputs = maincode.build_model_inputs(messages)
    model_inputs = {
        key: value.to(maincode.device)
        for key, value in model_inputs.items()
        if hasattr(value, "to")
    }
    input_length = model_inputs["input_ids"].shape[1]

    rng_seed = sample_seed_value(question_number, sample_index)
    torch.manual_seed(rng_seed)

    sampling_kwargs = {
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    if TOP_K:
        sampling_kwargs["top_k"] = TOP_K

    with torch.no_grad():
        outputs = maincode.model.generate(
            **model_inputs,
            max_new_tokens=SEED_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            **sampling_kwargs,
        )

    generated_token_ids = outputs.sequences[0, input_length:]
    answer = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
    features = generation_features(raw_step_logits(outputs), generated_token_ids)
    return answer, features, rng_seed


def judge_seed(gemini_model, question, answer):
    prompt = SEED_JUDGE_TEMPLATE.format(question=question, answer=answer)
    raw_text = call_gemini(gemini_model, prompt)
    label, reason = parse_seed_judgement(raw_text)
    return label, reason, raw_text


def main():
    questions = load_questions()
    processed_samples, answers_by_question = load_existing_seeds()

    question_items = list(questions.items())
    if MAX_QUESTIONS:
        question_items = question_items[:MAX_QUESTIONS]

    pending = [
        (number, question, sample_index)
        for number, question in question_items
        for sample_index in range(SAMPLES_PER_QUESTION)
        if (number, sample_index) not in processed_samples
    ]

    print(f"Test model: {MODEL_NAME}")
    print(f"Judge model: {JUDGE_MODEL_NAME}")
    print(f"Questions: {QUESTIONS_PATH.name} ({len(question_items)} in use)")
    print(f"Output: {SEEDS_PATH.name}")
    print(
        f"Sampling: temperature={TEMPERATURE}, top_p={TOP_P}"
        + (f", top_k={TOP_K}" if TOP_K else "")
        + f", {SAMPLES_PER_QUESTION} samples/question (base seed {BASE_SEED})"
    )
    print(
        f"Pending: {len(pending)} generations "
        f"({len(processed_samples)} already done for this model)"
    )

    if maincode.env_str("DRY_RUN", "") == "1":
        print("DRY_RUN=1 set; validation passed, exiting before model/API calls.")
        return

    if not pending:
        print("Nothing to do.")
        return

    gemini_model = setup_gemini()
    label_counts = Counter()
    duplicate_count = 0

    for index, (question_number, question, sample_index) in enumerate(pending, start=1):
        answer, features, rng_seed = generate_seed_answer(
            question,
            question_number,
            sample_index,
        )
        progress = f"[{index}/{len(pending)}] q{question_number}#{sample_index}"

        if not answer:
            print(f"{progress}: empty generation, skipping")
            continue

        seen_answers = answers_by_question.setdefault(question_number, set())
        is_duplicate = answer in seen_answers
        seen_answers.add(answer)

        if is_duplicate:
            # Recorded rather than dropped so the sample counts as done, but
            # flagged so the branching stage does not spend a full grid on a
            # seed identical to one it already has.
            duplicate_count += 1
            record = {
                "seed_schema_version": SEED_SCHEMA_VERSION,
                "question_number": question_number,
                "sample_index": sample_index,
                "question": question,
                "model_answer": answer,
                "model_name": MODEL_NAME,
                "duplicate_answer": True,
                "gemini_judgement": "Overall label: Not Hallucinating",
            }
            with open(SEEDS_PATH, "a") as seeds_file:
                seeds_file.write(json.dumps(record) + "\n")
            print(f"{progress}: duplicate of an earlier sample, not judged")
            continue

        label, reason, judge_raw = judge_seed(gemini_model, question, answer)
        label_counts[label] += 1

        record = {
            "seed_schema_version": SEED_SCHEMA_VERSION,
            "question_number": question_number,
            "sample_index": sample_index,
            "question": question,
            "model_answer": answer,
            "model_name": MODEL_NAME,
            "judge_model_name": JUDGE_MODEL_NAME,
            "max_new_tokens": SEED_MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "rng_seed": rng_seed,
            "duplicate_answer": False,
            # Consumed by maincode.py, which selects seeds whose judgement
            # starts with "Overall label: Hallucinating".
            "gemini_judgement": f"Overall label: {label}",
            "judge_reason": reason,
            "judge_raw": judge_raw,
        }
        record["seed_id"] = seed_identifier(record)
        if features:
            record.update({f"gen_{name}": value for name, value in features.items()})

        with open(SEEDS_PATH, "a") as seeds_file:
            seeds_file.write(json.dumps(record) + "\n")

        print(
            f"{progress}: {label}" + (f" - {reason[:80]}" if reason else "")
        )

    total = sum(label_counts.values())
    hallucinating = label_counts["Hallucinating"]
    print(f"\nJudged {total} answers: {hallucinating} hallucinating, {total - hallucinating} clean")
    if total:
        print(f"Hallucination rate: {hallucinating / total:.1%}")
    if duplicate_count:
        print(f"Skipped {duplicate_count} duplicate answer(s); raise TEMPERATURE for more variety.")
    print(f"Wrote {SEEDS_PATH}")
    print(f"\nNext: INPUT_PATH={SEEDS_PATH.name} TEST_MODEL={MODEL_NAME} python maincode.py")


if __name__ == "__main__":
    main()
