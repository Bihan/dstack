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


# --- Batch Search for Similar Issues ---
def search_similar_issues_batch(
    keyword: str, repo: str, exclude_id: int, original_title: str = "", original_body: str = ""
) -> List[Dict]:
    url = "https://api.github.com/search/issues"
    # sanitize and wrap keyword in quotes for GitHub search
    clean_keyword = keyword.replace("\n", " ").replace('"', "").strip()
    query = f'repo:{repo} is:issue "{clean_keyword}"'
    params = {"q": query, "per_page": 5}
    response = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    response.raise_for_status()

    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
    if remaining < 1:
        reset = int(response.headers.get("X-RateLimit-Reset", 0))
        time.sleep(max(reset - time.time(), 0) + 1)

    candidates = [i for i in response.json().get("items", []) if i.get("number") != exclude_id]
    if not candidates:
        return []

    prompt_lines = [
        "Compare these issues to the original:",
        "",
        f"Original Title: {original_title}",
        f"Original Body: {original_body}",
        "",
        "Candidate issues:",
    ]
    for idx, issue in enumerate(candidates, start=1):
        prompt_lines += [
            f"Issue {idx} (#{issue['number']}):",
            f"  Title: {issue['title']}",
            f"  Body: {issue.get('body', '')}",
            "",
        ]
    prompt_lines.append(
        "Which of these is most likely the same as the original issue? Reply with the issue number only."
    )
    batch_prompt = "\n".join(prompt_lines)

    # Single LLM call
    content = generate_llm_step(batch_prompt)
    m = re.search(r"\b(\d+)\b", content)
    picked = int(m.group(1)) if m else None

    results = []
    for issue in candidates:
        results.append(
            {
                "issue_id": issue["number"],
                "title": issue["title"],
                "state": issue.get("state", ""),
                "url": issue.get("html_url", ""),
                "created_at": issue.get("created_at", ""),
                "labels": [lbl.get("name", "") for lbl in issue.get("labels", [])],
                "related": (issue["number"] == picked),
            }
        )
    return results


# --- LLM Setup ---
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# def generate_llm_step(prompt: str) -> str:
#     resp = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are an intelligent agent that thinks before acting. Use THOUGHT and ACTION format.",
#             },
#             {"role": "user", "content": prompt},
#         ],
#         temperature=0.3,
#     )
#     return resp.choices[0].message.content

# --- vLLM Setup ---
# vLLM local API endpoint
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"


def generate_llm_step(prompt: str) -> str:
    """
    Send a prompt to the local vLLM endpoint and return the assistant content string.
    """
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
    data = resp.json()
    # Extract the assistant's reply content
    return data["choices"][0]["message"]["content"]


# --- Build a Single Trajectory with multiple unique keyword iterations ---
def build_trajectory(target_issue: Dict, repo: str, max_iters: int = MAX_KEYWORD_ITERS) -> Dict:
    title = target_issue["title"]
    issue_id = target_issue["number"]
    instruction = f"Check if the issue '{title}' already exists. If not, create a GitHub issue."

    steps = []
    seen_norm_keywords = set()
    related_ids: List[int] = []

    for i in range(1, max_iters + 1):
        # 1) generate a new keyword
        prompt = f"Iteration {i}: Given the issue title '{title}', suggest a concise search keyword in quotes."
        llm_kw = generate_llm_step(prompt)
        keyword = extract_keyword_from_llm_output(llm_kw)
        # normalize: strip backticks, quotes, lowercase
        norm = re.sub(r"[`\"']", "", keyword).lower().strip()

        # 2) ensure uniqueness on normalized form
        if norm in seen_norm_keywords:
            alt_prompt = f"Iteration {i}: You already suggested '{keyword}'. Please suggest a different concise search keyword in quotes."
            llm_kw = generate_llm_step(alt_prompt)
            keyword = extract_keyword_from_llm_output(llm_kw)
            norm = re.sub(r"[`\"']", "", keyword).lower().strip()
            if norm in seen_norm_keywords:
                continue
        seen_norm_keywords.add(norm)

        # 3) perform batch search
        thought = f"Thought {i}: search using keyword '{keyword}'"
        act = f"Act {i}: SearchGitIssueTool(title:{keyword})"
        observations = search_similar_issues_batch(
            keyword,
            repo,
            exclude_id=issue_id,
            original_title=title,
            original_body=target_issue.get("body", ""),
        )
        steps.append({"thought": thought, "act": act, "observation": observations})

        # 4) collect any related IDs and break early
        for item in observations:
            if item.get("related"):
                related_ids.append(item["issue_id"])
        if related_ids:
            break

    # --- decide final action ---
    if related_ids:
        final_action = f"GetRelatedIssueIdsTools(issue_ids:{related_ids})"
    else:
        # desc = target_issue.get("body", " ").replace("\n", " ")
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
