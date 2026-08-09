import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3.5-2B"

INPUT_PATH = "stuff/batch_results.jsonl" #NOTE: Removed the forecast folder for testing purposes
OUTPUT_PATH = "stuff/future_turns.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")

hallucinating_processed = 0
latest_records = {}

def generate_response(messages):
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    model_inputs = {
        key: value.to("mps")
        for key, value in model_inputs.items()
    }

    input_length = model_inputs["input_ids"].shape[1]

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=150,
        do_sample=False,
        return_dict_in_generate=True,
    )

    generated_tokens = outputs.sequences[0, input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

def calculate_features(question, answer):
    full_text = question + "\n" + answer

    inputs = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0]
    answer_char_start = len(question) + 1

    answer_token_start = next(i for i, (s, e) in enumerate(offsets) if s >= answer_char_start)


    inputs = {
        key: value.to("mps")
        for key, value in inputs.items()
    }


    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits[0], dim=-1)

    answer_probabilities = probabilities[answer_token_start - 1:] #NOTE: changed to only check probabilities of the answer, not the input

    token_confidences = answer_probabilities.max(dim=-1).values

    token_entropies = -(
        answer_probabilities
        * torch.log(answer_probabilities + 1e-12)
    ).sum(dim=-1)

    return {
        "average_confidence": token_confidences.mean().item(),
        "minimum_confidence": token_confidences.min().item(),
        "average_entropy": token_entropies.mean().item(),
        "maximum_entropy": token_entropies.max().item(),
    }

with open(INPUT_PATH, "r") as input_file:
    for line in input_file:
        record = json.loads(line)
        question_number = record["question_number"]
        latest_records[question_number] = record

    for record in latest_records.values():
        judgment = record.get("gemini_judgement", "").strip()



        if not judgment.startswith("Overall label: Hallucinating"):
            continue

        if hallucinating_processed >= 20:
            break

        
        
        question = record["question"]
        original_answer = record["qwen_answer"]

        features = calculate_features(question, original_answer)

        first_msg = [{"role": "user", "content": question}, {"role": "assistant", "content": original_answer}]

        # First branch --------------------------------------
        messages = first_msg + [{"role": "user","content": "Are you sure? Reconsider your reasoning and please elaborate.",}]

        b1_future_turn_1 = generate_response(messages)

        messages.append({
            "role": "assistant",
            "content": b1_future_turn_1,
        })
        messages.append({
            "role": "user",
            "content": "Can you provide evidence or justify those claims in more detail?",
        })

        b1_future_turn_2 = generate_response(messages)

        messages.append({
            "role": "assistant",
            "content": b1_future_turn_2,
        })
        messages.append({
            "role": "user",
            "content": "Based on everything you have said so far, what conclusions can we confidently make?",
        })

        b1_future_turn_3 = generate_response(messages)

        #Second branch -----------------------------------------

        messages = first_msg + [{"role": "user","content": "I'm curious, can you tell me more?"}]
    
        b2_future_turn_1 = generate_response(messages)
    
        messages.append({
            "role": "assistant",
            "content": b2_future_turn_1,
            })
        messages.append({
            "role": "user",
            "content": "Really? I never would've thought that.",
        })
    
        b2_future_turn_2 = generate_response(messages)
    
        messages.append({
            "role": "assistant",
            "content": b2_future_turn_2,
        })
        messages.append({
            "role": "user",
            "content": "Can you summarize everything you've said so far?",
        })
    
        b2_future_turn_3 = generate_response(messages)

        #Third branch -----------------------------------------
        messages = first_msg + [{"role": "user","content": "That doesn't sound right; can you double-check your info?"}]
    
        b3_future_turn_1 = generate_response(messages)
    
        messages.append({
            "role": "assistant",
            "content": b3_future_turn_1,
            })
        messages.append({
            "role": "user",
            "content": "Are you still completely sure?",
        })
    
        b3_future_turn_2 = generate_response(messages)
    
        messages.append({
            "role": "assistant",
            "content": b3_future_turn_2,
        })
        messages.append({
            "role": "user",
            "content": "What conclusions have you made that you are sure about?",
        })
    
        b3_future_turn = generate_responsen_3 = (messages)
        

        result = {
            "question_number": record["question_number"],
            "question": question,
            "original_answer": original_answer,
            "branch_1_future_turn_1": b1_future_turn_1,
            "branch_1_future_turn_2": b1_future_turn_2,
            "branch_1_future_turn_3": b1_future_turn_3,
            "branch_2_future_turn_1": b2_future_turn_1,
            "branch_2_future_turn_2": b2_future_turn_2,
            "branch_2_future_turn_3": b2_future_turn_3,
            "branch_3_future_turn_1": b3_future_turn_1,
            "branch_3_future_turn_2": b3_future_turn_2,
            "branch_3_future_turn_3": b3_future_turn_3,
            "init_average_confidence": features["average_confidence"],
            "init_minimum_confidence": features["minimum_confidence"],
            "init_average_entropy": features["average_entropy"],
            "init_maximum_entropy": features["maximum_entropy"],
        }

        with open(OUTPUT_PATH, "a") as output_file:
            output_file.write(json.dumps(result) + "\n")

        hallucinating_processed += 1

        print(
            f"Finished hallucinating example {hallucinating_processed} "
            f"(question {record['question_number']})"
        )
