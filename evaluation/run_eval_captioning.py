"""
This script evaluates a caption submission through a three-stage GPT pipeline: 
1) Event Splitting, 2) Event Matching, and 3) Relationship Classification. 
It then computes scores based on the alignment between predicted and reference event relationships.

Usage:
    python evaluate_submission.py \
        --submission_file path/to/submission.json \
        --metadata_file path/to/metadata.json \
        --output_dir ./results \
        --model gpt-4o \
        --api_key YOUR_OPENAI_API_KEY

Outputs:
    - Three-stage evaluation result (JSON)
    - Scored result with TUNA-CAP scores (JSON)
    - Score summary (JSON)
"""
import os
import json
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from tool.prompt import PROMPT_SPLIT_EVENT, PROMPT_MATCH_EVENT, PROMPT_CLASSIFY_RELATION
from tool.gpt_client import call_gpt
from tool.scorer import CaptioningScorer
from tool.utils import parse_string_to_obj, check_match_ids, print_green, print_red


# Load metadata: index -> metadata info
def load_metadata(metadata_file: str) -> Dict[str, dict]:
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return {item['index']: item for item in metadata}

# Load submission: index -> caption
def load_submission(submission_file: str) -> Dict[str, str]:
    with open(submission_file, 'r', encoding='utf-8') as f:
        submission_data = json.load(f)
    return submission_data


def build_split_question(caption: str) -> str:
    return PROMPT_SPLIT_EVENT.format(caption=caption).strip()


def build_match_question(candidate_events: list, reference_events: list) -> str:
    args = {
        "candidate_events": [{"id": i + 1, "event": e} for i, e in enumerate(candidate_events)],
        "reference_events": [{"id": i + 1, "event": e} for i, e in enumerate(reference_events)],
    }
    return PROMPT_MATCH_EVENT.format(**args).strip()


def build_classify_question(match_data) -> str:
    args = {
        "match_data": [
            {"candidate_event": d["candidate_event"], "visual_elements": d["visual_elements"]}
            for d in match_data
        ]
    }
    return PROMPT_CLASSIFY_RELATION.format(**args).strip()


def evaluate_submission_pipeline(submission_file: str, metadata_map: Dict[str, dict], output_file: str, model: str, headers: dict, retry: int = 3):
    submission = load_submission(submission_file)

    results = []

    for idx, caption in tqdm(submission.items(), desc="Evaluating"):
        if idx not in metadata_map:
            print(f"Skip index not found in metadata, index: {idx}")
            continue
        meta = metadata_map[idx]
        result = {"index": idx, "candidate_caption": caption}

        # ==== Step 1: Split ====
        split_prompt = build_split_question(caption)
        for _ in range(retry):
            success, _, split_answer = call_gpt(model, [{"role": "user", "content": split_prompt}], headers)
            if not success:
                continue
            candidate_events = parse_string_to_obj(split_answer)
            if candidate_events:
                break
        else:
            print_red(f"[{idx}] Failed to parse format in `Split` stage. Retry numbers: {retry}.\nRaw split_answer: {split_answer}")
            continue
        result["candidate_events"] = candidate_events

        # ==== Step 2: Match ====
        reference_events = [e["event"] for e in meta["events"]]
        match_prompt = build_match_question(candidate_events, reference_events)
        for _ in range(retry):
            success, _, match_answer = call_gpt(model, [{"role": "user", "content": match_prompt}], headers)
            if not success:
                continue
            match_ids = parse_string_to_obj(match_answer)
            if match_ids and check_match_ids(match_ids):
                break
        else:
            print_red(f"[{idx}] Failed to parse format in `Match` stage. Retry numbers: {retry}.\nRaw match_answer: {match_answer}")
            continue
        result["match_ids"] = match_ids

        # ==== Step 3: Classify ====
        match_data = [
            {"candidate_event": [], "reference_event": ref, "visual_elements": [v["content"] for v in ref["visual_elements"]]} 
            for ref in meta["events"]
        ]
        for cid, rid in match_ids:
            if rid is not None:
                match_data[rid - 1]["candidate_event"].append(candidate_events[cid - 1])
        for md in match_data:
            md["candidate_event"] = ' '.join(md["candidate_event"])
        classify_prompt = build_classify_question(match_data)

        for _ in range(retry):
            success, _, classify_answer = call_gpt(model, [{"role": "user", "content": classify_prompt}], headers)
            if not success:
                continue
            try:
                relationship = parse_string_to_obj(classify_answer)
                if relationship:
                    break
            except Exception:
                continue
        else:
            print_red(f"[{idx}] Failed to parse format in `Classify` stage. Retry numbers: {retry}.\nRaw classify_answer: {classify_answer}")
            continue
        result["relationship"] = relationship

        results.append(result)

    # Save three-stage result for scoring
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print_green("Three-stage result is saved to:", scored_file)

    return results


# Scoring function to apply CaptioningScorer to each evaluated record
def score_results(results: List[Dict], metadata_map: Dict[str, dict], scored_file: str, summary_file: str):
    scorer = CaptioningScorer()
    scored_results = []

    for item in tqdm(results, desc="Scoring"):
        idx = item["index"]
        # Skip if metadata missing for the index
        if idx not in metadata_map:
            print(f"[{idx}] Metadata missing, skipping scoring.")
            continue

        gt_events = metadata_map[idx]["events"]
        pred_relationship = item["relationship"]
        if len(gt_events) != len(pred_relationship):
            print(f"[{idx}] Mismatch between number of predicted and reference events. gt_events: {len(gt_events)}, pred_relationship: {len(pred_relationship)}.")
            continue

        # Compute the score of current instance
        score = scorer.score_instance(gt_events, pred_relationship)
        item["score"] = score
        scored_results.append(item)

        # Used to compute total average score
        category_tags = metadata_map[idx]["visual_characteristic"].split(",")  # Multiple Tags
        scorer.update_totals(score, category_tags)

    with open(scored_file, 'w', encoding='utf-8') as f:
        json.dump(scored_results, f, ensure_ascii=False, indent=2)
    print_green("Scored result is saved to:", scored_file)
    
    average_scores = scorer.compute_averages()
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(average_scores, f, ensure_ascii=False, indent=2)
    print_green("TUNA-CAP score is saved to:", summary_file)


def load_or_resume_results(submission_file: str,
                           metadata_map: Dict[str, dict],
                           result_file: str,
                           model: str,
                           headers: dict) -> List[Dict]:
    if not os.path.exists(result_file):
        results = evaluate_submission_pipeline(submission_file, metadata_map, result_file, model, headers)
        return results

    # Load existing results
    with open(result_file, 'r', encoding='utf-8') as f:
        existing_results = json.load(f)
    existing_index_set = {item["index"] for item in existing_results}
    print(f"[Info] Loaded {len(existing_results)} existing results.")

    # Find missing indices
    all_index_set = set(metadata_map.keys())
    missing_index_set = all_index_set - existing_index_set
    print(f"[Info] Missing {len(missing_index_set)} indices. Start recovery.")

    if len(missing_index_set) == 0:
        return existing_results

    # Re-evaluate the missing part
    partial_metadata_map = {idx: metadata_map[idx] for idx in missing_index_set}
    new_results = evaluate_submission_pipeline(submission_file, partial_metadata_map, 
                                               result_file.replace('_result.json', 'temp_new_result.json'),
                                               model, headers)

    # Merge and save all results
    all_results = existing_results + new_results
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[Info] Completed recovery. Total results: {len(all_results)}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_file", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--api_key", type=str, required=True)
    args = parser.parse_args()

    submission_name = os.path.splitext(os.path.basename(args.submission_file))[0]
    os.makedirs(f"{args.output_dir}/{submission_name}", exist_ok=True)
    result_file = f"{args.output_dir}/{submission_name}/{submission_name}_result.json"
    scored_file = f"{args.output_dir}/{submission_name}/{submission_name}_scored.json"
    summary_file = f"{args.output_dir}/{submission_name}/{submission_name}_score_summary.json"

    headers = {"Authorization": f"Bearer {args.api_key}"}
    metadata_map = load_metadata(args.metadata_file)

    # Load existing results if available, or perform inference on missing entries only
    results = load_or_resume_results(args.submission_file, metadata_map, result_file, args.model, headers)
    # Score all results (per-instance and aggregated multi-dimensional averages)
    score_results(results, metadata_map, scored_file, summary_file)

