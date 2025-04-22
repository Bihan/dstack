import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List

import requests

# --- Configuration ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
REPO = "pytorch/pytorch"
OWNER, REPO_NAME = REPO.split("/")
HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

# Number of keyword iterations
MAX_KEYWORD_ITERS = 3
# Timeout for HTTP requests (in seconds)
TIMEOUT = 15000
SAVE_DIR = "/root/.cache/huggingface/hub/data"


# --- GitHub Fetch ---
def fetch_issues(repo: str, state: str = "all", max_pages: int = 5) -> List[Dict]:
    all_issues = []
    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": state, "per_page": 100, "page": page}
        response = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            break

        remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        if remaining < 1:
            reset = int(response.headers.get("X-RateLimit-Reset", 0))
            time.sleep(max(reset - time.time(), 0) + 1)

        data = response.json()
        issues = [i for i in data if "pull_request" not in i]
        if not issues:
            break
        all_issues.extend(issues)

    with open("fetched_issues.jsonl", "w") as fout:
        for issue in all_issues:
            fout.write(json.dumps(issue) + "\n")
    return all_issues


# --- Keyword Extraction ---
def extract_keyword_from_llm_output(text: str, default: str = "") -> str:
    """
    Extract the last quoted keyword. If none found, fallback to `default` or first five words.
    """
    matches = re.findall(r'"([^\"]+)"', text)
    if matches:
        return matches[-1]
    if default:
        return default
    words = text.strip().split()
    return " ".join(words[:5])


VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"


# --- LLM Helpers ---
def generate_llm_step(prompt: str) -> str:
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an intelligent agent that thinks before acting. Use THOUGHT and ACTION format.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(
        VLLM_API_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# New helper to generate reasoning


def generate_reason(prompt: str) -> str:
    """
    Ask the LLM to explain a choice or classification and return concise reason.
    """
    content = generate_llm_step(prompt)
    # strip any quotes/extra formatting
    return content.strip().replace("\n", " ")


# --- Batch Search for Similar Issues ---
def search_similar_issues_batch(
    keyword: str, repo: str, exclude_id: int, original_title: str = "", original_body: str = ""
) -> List[Dict]:
    url = "https://api.github.com/search/issues"
    clean_keyword = keyword.replace("\n", " ").replace('"', "").strip()
    query = f'repo:{repo} is:issue "{clean_keyword}"'
    params = {"q": query, "per_page": 5}
    response = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    response.raise_for_status()

    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
    if remaining < 1:
        reset = int(response.headers.get("X-RateLimit-Reset", 0))
        time.sleep(max(reset - time.time(), 0) + 1)

    items = response.json().get("items", [])
    candidates = [i for i in items if i.get("number") != exclude_id]
    if not candidates:
        return []

    # build prompt for LLM to pick related issue
    compare_lines = [
        "Compare these issues to the original:",
        f"Original Title: {original_title}",
        f"Original Body: {original_body}",
        "Candidate issues:",
    ]
    for idx, issue in enumerate(candidates, start=1):
        compare_lines += [
            f"Issue {idx} (#{issue['number']}):",
            f"  Title: {issue['title']}",
            f"  Body: {issue.get('body', '')}",
        ]
    compare_lines.append(
        "Which of these is most likely the same as the original issue? Reply with the issue number only."
    )
    picked_resp = generate_llm_step("\n".join(compare_lines))
    m = re.search(r"\b(\d+)\b", picked_resp)
    picked = int(m.group(1)) if m else None

    observations = []
    for issue in candidates:
        is_related = issue["number"] == picked
        # generate reasoning for related decision
        reason_prompt = (
            f"Explain why issue #{issue['number']} ('{issue['title']}') is "
            + ("related" if is_related else "not related")
            + f" to the original title '{original_title}'. Provide a concise reason."
        )
        reason_text = generate_reason(reason_prompt)

        observations.append(
            {
                "issue_id": issue["number"],
                "title": issue["title"],
                "state": issue.get("state", ""),
                "url": issue.get("html_url", ""),
                "created_at": issue.get("created_at", ""),
                "labels": [lbl.get("name", "") for lbl in issue.get("labels", [])],
                "related": is_related,
                "reason": reason_text,
            }
        )
    return observations


# --- Build a Single Trajectory with reasoning ---
def build_trajectory(target_issue: Dict, repo: str, max_iters: int = MAX_KEYWORD_ITERS) -> Dict:
    title = target_issue["title"]
    issue_id = target_issue["number"]
    instruction = f"Check if the issue '{title}' already exists. If not, create a GitHub issue."

    steps = []
    seen_norm_keywords = set()
    related_ids: List[int] = []

    for i in range(1, max_iters + 1):
        # 1) generate a new keyword
        prompt_kw = f"Iteration {i}: Given the issue title '{title}', suggest a concise search keyword in quotes."
        llm_kw = generate_llm_step(prompt_kw)
        keyword = extract_keyword_from_llm_output(llm_kw)
        norm = re.sub(r"[`\"']", "", keyword).lower().strip()

        # ensure uniqueness
        if norm in seen_norm_keywords:
            alt_prompt = f"Iteration {i}: You already suggested '{keyword}'. Please suggest a different concise search keyword in quotes."
            llm_kw = generate_llm_step(alt_prompt)
            keyword = extract_keyword_from_llm_output(llm_kw)
            norm = re.sub(r"[`\"']", "", keyword).lower().strip()
            if norm in seen_norm_keywords:
                continue
        seen_norm_keywords.add(norm)

        # 1.b) generate reasoning for this keyword choice
        reason_prompt = f"Explain why '{keyword}' is a good keyword to search for issue '{title}'. Provide a concise reason."
        reason_text = generate_reason(reason_prompt)

        # 2) perform batch search
        thought = f"Thought {i}: search using keyword '{keyword}' because {reason_text}"
        act = f"Act {i}: SearchGitIssueTool(title:{keyword})"
        observations = search_similar_issues_batch(
            keyword,
            repo,
            exclude_id=issue_id,
            original_title=title,
            original_body=target_issue.get("body", ""),
        )
        steps.append({"thought": thought, "act": act, "observation": observations})

        # collect any related IDs and break early
        for item in observations:
            if item.get("related"):
                related_ids.append(item["issue_id"])
        if related_ids:
            break

    # --- decide final action ---
    if related_ids:
        final_action = f"GetRelatedIssueIdsTools(issue_ids:{related_ids})"
    else:
        final_action = (
            f"CreateIssueTool(title:'{title}', environment:'...',"
            f" description:'...', steps_to_reproduce:'...')"
        )

    return {
        "instruction": instruction,
        "issue_id": issue_id,
        "steps": steps,
        "final_action": final_action,
        "reward": 1.0,
    }


# --- Generate Real Trajectories ---
def generate_real_trajectories(repo: str, n: int = 5):
    print("⏳ Fetching issues...")
    issues = fetch_issues(repo)
    selected = random.sample(issues, min(n, len(issues)))
    print(f"✅ Generating {len(selected)} trajectories...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trajectories = [build_trajectory(issue, repo) for issue in selected]
    with open(f"{SAVE_DIR}/real_github_trajectories_{timestamp}.json", "w") as f:
        json.dump(trajectories, f, indent=2)
    print("🎯 Saved to real_github_trajectories.json")


if __name__ == "__main__":
    generate_real_trajectories(REPO, n=2)
