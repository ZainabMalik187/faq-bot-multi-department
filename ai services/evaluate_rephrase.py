"""
evaluate_rephrase.py — Rephrase-based evaluation for the FAQ bot.

How it works:
  1. Load all FAQs from data/faq.json.
  2. Randomly pick N questions (default 10).
  3. Use Groq Llama-3 to rephrase each question in casual language.
  4. Send the rephrased question to the FAQ bot API.
  5. Use Groq Llama-3 as a judge to check whether the bot's answer
     semantically matches the original FAQ answer.
  6. Print a detailed report and save results to a JSON file.

Run:
  python evaluate_rephrase.py            # 10 random questions
  python evaluate_rephrase.py --count 20 # 20 random questions
  python evaluate_rephrase.py --all      # every FAQ entry

Requires the bot server to be running:
  python -m uvicorn main:app --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from groq import Groq
from google import genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

# API_URL = os.environ.get("FAQ_BOT_URL", "http://127.0.0.1:8000/chat")

API_URL = "http://127.0.0.1:8000/chat"
# Load all keys from environment
GROQ_API_KEYS = [
    os.environ.get(f"GROQ_API_KEY{i}")
    for i in (1, 2, 3)
    if os.environ.get(f"GROQ_API_KEY{i}")
]
if not GROQ_API_KEYS:
    default_key = os.environ.get("GROQ_API_KEY", "")
    if default_key:
        GROQ_API_KEYS = [default_key]

_current_key_idx = 0

def get_current_key() -> str:
    if not GROQ_API_KEYS:
        return ""
    return GROQ_API_KEYS[_current_key_idx % len(GROQ_API_KEYS)]

def rotate_key() -> str:
    global _current_key_idx
    if not GROQ_API_KEYS:
        return ""
    _current_key_idx += 1
    new_key = get_current_key()
    print(f"    [INFO] Rotating Groq API Key to index {_current_key_idx % len(GROQ_API_KEYS)}")
    return new_key

GROQ_MODEL = "llama-3.1-8b-instant"
FAQ_PATH = Path(__file__).resolve().parent / "data" / "faq.json"
REPORT_DIR = Path(__file__).resolve().parent / "eval_reports"

# Delay between API calls to avoid Groq rate limits (seconds)
DELAY_BETWEEN_CALLS = 5


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------
def rephrase_question(original_question: str) -> str:
    """Ask Groq to rephrase the FAQ question in casual, everyday language."""
    prompt = (
        "Rephrase the following company FAQ question using casual, everyday language, "
        "as a real employee would ask it. Keep the same core meaning and scope (do NOT "
        "change the target audience, product type, or department context). "
        "Do NOT copy the original wording. Return ONLY the rephrased question, "
        "nothing else.\n\n"
        f"Original: {original_question}\n"
        "Rephrased:"
    )
    max_attempts = max(1, len(GROQ_API_KEYS))
    for attempt in range(max_attempts):
        try:
            client = Groq(api_key=get_current_key(), timeout=15.0)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [WARN] Rephrase attempt {attempt+1}/{max_attempts} failed ({e}). Rotating key...")
            if attempt < max_attempts - 1:
                rotate_key()
                time.sleep(1)
    print(f"    [WARN] Rephrase failed all retries, using original question.")
    return original_question


def judge_answer(original_answer: str, bot_answer: str) -> tuple[bool, str]:
    """
    Ask Groq to judge whether the bot's answer conveys the same meaning
    as the original FAQ answer.

    Returns (passed: bool, reason: str).
    """
    prompt = (
        "You are an evaluation judge. Compare the two answers below and "
        "decide if they convey the SAME core information.\n\n"
        "Rules:\n"
        "- Minor wording differences are OK.\n"
        "- Extra helpful details in the bot answer are OK.\n"
        "- The bot answer must NOT contradict the original.\n"
        "- The bot answer must cover the key facts of the original.\n\n"
        f'Original FAQ answer:\n"{original_answer}"\n\n'
        f'Bot answer:\n"{bot_answer}"\n\n'
        "Respond in this exact format (two lines only):\n"
        "VERDICT: PASS or FAIL\n"
        "REASON: one-sentence explanation"
    )
    max_attempts = max(1, len(GROQ_API_KEYS))
    for attempt in range(max_attempts):
        try:
            client = Groq(api_key=get_current_key(), timeout=15.0)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            text = response.choices[0].message.content.strip()

            # Parse verdict
            passed = "PASS" in text.split("\n")[0].upper()
            reason = text.split("REASON:")[-1].strip() if "REASON:" in text else text
            return passed, reason
        except Exception as e:
            print(f"    [WARN] Judge attempt {attempt+1}/{max_attempts} failed ({e}). Rotating key...")
            if attempt < max_attempts - 1:
                rotate_key()
                time.sleep(1)
    return False, "Judge API error after retries"


# ── DEPT VALIDATION: frontend department vs question department ───
def ask_bot(query: str, department: str) -> dict | None:
    """Send a query to the FAQ bot and return the response dict."""
    try:
        resp = requests.post(API_URL, json={"department": department, "text": query}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    [ERROR] Bot API failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Grounding and Memory Isolation Tests
# ---------------------------------------------------------------------------
def test_grounding_strictness() -> bool:
    """
    Submits a vague query to check if the bot hallucinates generic details
    or follows grounding instructions strictly.
    """
    print("Running test_grounding_strictness...")
    # ── DEPT VALIDATION: frontend department vs question department ───
    payload = {"department": "HR", "text": "do you know about hr of pel", "session_id": "test_grounding_session"}
    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "").lower()
        print(f"  Query : {payload['text']}")
        print(f"  Answer: {data.get('answer')}")
        
        # Check for forbidden generic phrases
        forbidden = [
            "responsible for various tasks",
            "among other things",
            "and more",
            "feel free to ask",
            "employee onboarding",
            "benefits administration",
            "performance management"
        ]
        
        for phrase in forbidden:
            if phrase in answer:
                print(f"  [FAIL] Answer contains hallucinated/generic phrase: '{phrase}'")
                return False
                
        # Verify it contains the contact statement or records statement.
        expected_fallback = (
            "I don't have information about that. Please contact our "
            "support team directly, or try asking about HR, IT, Sales, "
            "Finance, or Customer Support topics."
        ).lower()
        if "contact the hr department" not in answer and expected_fallback not in answer:
            print(f"  [FAIL] Answer does not contain the mandatory redirection/contact statement.")
            return False
            
        print("  [PASS] Grounding strictness test passed.")
        return True
    except Exception as e:
        print(f"  [FAIL] Request failed: {e}")
        return False


def test_memory_isolation() -> bool:
    """
    Checks that the session memory isolates unrelated topics/queries
    and does not leak conversation history across turns.
    """
    print("Running test_memory_isolation...")
    session_id = "test_memory_session"
    
    # Turn 1: HR specific query about pursuing higher education
    q1 = "Can I pursue higher education while working at PEL?"
    print(f"  Turn 1 Query: {q1}")
    # ── DEPT VALIDATION: frontend department vs question department ───
    try:
        resp1 = requests.post(API_URL, json={"department": "HR", "text": q1, "session_id": session_id}, timeout=60)
        resp1.raise_for_status()
        data1 = resp1.json()
        print(f"  Turn 1 Ans  : {data1.get('answer')}")
    except Exception as e:
        print(f"  [FAIL] Turn 1 request failed: {e}")
        return False
        
    time.sleep(DELAY_BETWEEN_CALLS)
    
    # Turn 2: Query an unrelated department/topic to trigger memory check/reset (IT support)
    q2 = "how do I reset my password?"
    print(f"  Turn 2 Query: {q2}")
    # ── DEPT VALIDATION: frontend department vs question department ───
    try:
        resp2 = requests.post(API_URL, json={"department": "IT", "text": q2, "session_id": session_id}, timeout=60)
        resp2.raise_for_status()
        data2 = resp2.json()
        print(f"  Turn 2 Ans  : {data2.get('answer')}")
    except Exception as e:
        print(f"  [FAIL] Turn 2 request failed: {e}")
        return False
        
    time.sleep(DELAY_BETWEEN_CALLS)

    # Turn 3: Ask a general query about HR that shouldn't leak Turn 1
    q3 = "do you know about hr of pel"
    print(f"  Turn 3 Query: {q3}")
    # ── DEPT VALIDATION: frontend department vs question department ───
    try:
        resp3 = requests.post(API_URL, json={"department": "HR", "text": q3, "session_id": session_id}, timeout=60)
        resp3.raise_for_status()
        data3 = resp3.json()
        ans3 = data3.get("answer", "")
        print(f"  Turn 3 Ans  : {ans3}")
        
        # Check if Turn 1 context leaked into Turn 3 response
        leaked_keywords = ["noc", "education", "evening", "weekend", "classes", "previously asked", "recap"]
        ans3_lower = ans3.lower()
        for kw in leaked_keywords:
            if kw in ans3_lower:
                print(f"  [FAIL] Leak detected! Turn 3 answer contains context from Turn 1: '{kw}'")
                return False
                
        print("  [PASS] Memory isolation test passed.")
        return True
    except Exception as e:
        print(f"  [FAIL] Turn 3 request failed: {e}")
        return False


def test_out_of_scope_and_formatting() -> bool:
    """
    Checks that out-of-scope questions are correctly classified as "Unknown" department
    with the generic fallback message, and that responses do not contain the department name
    as a trailing duplicated word.
    """
    print("Running test_out_of_scope_and_formatting...")
    # ── DEPT VALIDATION: frontend department vs question department ───
    payload = {"department": "HR", "text": "who is the CEO of PEL", "session_id": "test_out_of_scope_session"}
    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        dept = data.get("department", "")
        print(f"  Query : {payload['text']}")
        print(f"  Answer: {answer}")
        print(f"  Dept  : {dept}")
        
        # Assertion 1: Out-of-scope question gets "Unknown" department
        if dept != "Unknown":
            print(f"  [FAIL] Expected department to be 'Unknown', but got '{dept}'")
            return False
            
        # Assertion 2: Answer matches GENERIC_FALLBACK_MESSAGE
        expected_fallback = (
            "I don't have information about that. Please contact our "
            "support team directly, or try asking about HR, IT, Sales, "
            "Finance, or Customer Support topics."
        )
        if answer != expected_fallback:
            print(f"  [FAIL] Expected answer to be generic fallback, but got '{answer}'")
            return False
            
        print("  [PASS] Out-of-scope check passed.")
        
        # Now test a valid query to ensure no duplicate trailing department word
        # ── DEPT VALIDATION: frontend department vs question department ───
        payload_valid = {"department": "IT", "text": "how do I connect to the corporate wifi?", "session_id": "test_out_of_scope_session"}
        resp_valid = requests.post(API_URL, json=payload_valid, timeout=60)
        resp_valid.raise_for_status()
        data_valid = resp_valid.json()
        ans_valid = data_valid.get("answer", "").strip()
        dept_valid = data_valid.get("department", "").strip()
        print(f"  Valid Query : {payload_valid['text']}")
        print(f"  Valid Answer: {ans_valid}")
        print(f"  Valid Dept  : {dept_valid}")
        
        if dept_valid:
            # Check if answer ends with the department word, e.g. "IT"
            # Split answer into words and check the last word
            words = [w.strip(".,!?\"'") for w in ans_valid.split()]
            if words and words[-1] == dept_valid:
                print(f"  [FAIL] Answer ends with duplicated trailing department name '{dept_valid}'")
                return False
                
        print("  [PASS] Formatting duplicate-department checks passed.")
        return True
    except Exception as e:
        print(f"  [FAIL] Out-of-scope and formatting test failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Rephrase-based FAQ bot evaluation")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of random FAQs to test (default: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Test every FAQ entry (ignores --count)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--port", type=int, default=None,
                        help="Override the bot API port (e.g. --port 8001)")
    args = parser.parse_args()

    # Allow port override
    global API_URL
    if args.port:
        API_URL = f"http://127.0.0.1:{args.port}/chat"

    if not get_current_key():
        print("[WARN] No Groq API keys found.")
        print("       Rephrasing and judging will fail.")
        print()

    # Load FAQs
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    if not faqs:
        print("No FAQs found in faq.json!")
        sys.exit(1)

    # Pick questions
    if args.all:
        selected = faqs
    else:
        if args.seed is not None:
            random.seed(args.seed)
        selected = random.sample(faqs, min(args.count, len(faqs)))

    total = len(selected)
    start_time = time.time()

    print("=" * 72)
    print("  FAQ Bot — Rephrase Evaluation Report")
    print("=" * 72)
    print(f"  API:       {API_URL}")
    print(f"  FAQs pool: {len(faqs)}")
    print(f"  Testing:   {total} questions")
    print(f"  Seed:      {args.seed or 'random'}")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    print()

    results = []

    for i, faq in enumerate(selected, start=1):
        faq_id = faq["id"]
        department = faq["department"]
        original_q = faq["question"]
        original_a = faq["answer"]

        print(f"[{i:02d}/{total:02d}] FAQ {faq_id} ({department})")
        print(f"  Original Q : {original_q}")

        # Step 1 — Rephrase
        rephrased_q = rephrase_question(original_q)
        print(f"  Rephrased Q: {rephrased_q}")
        time.sleep(DELAY_BETWEEN_CALLS)

        # ── DEPT VALIDATION: frontend department vs question department ───
        # Step 2 — Ask the bot (passing the actual department)
        bot_response = ask_bot(rephrased_q, department)
        if bot_response is None:
            results.append({
                "faq_id": faq_id,
                "department": department,
                "original_q": original_q,
                "original_a": original_a,
                "rephrased_q": rephrased_q,
                "bot_answer": None,
                "bot_dept": None,
                "passed": False,
                "reason": "Bot API error",
            })
            print(f"  Bot Answer : [API ERROR]")
            print(f"  -> FAIL (API error)\n")
            continue

        bot_answer = bot_response.get("answer", "")
        bot_dept = bot_response.get("department", "")
        print(f"  Bot Answer : {bot_answer[:120]}{'...' if len(bot_answer) > 120 else ''}")
        print(f"  Bot Dept   : {bot_dept}")
        time.sleep(DELAY_BETWEEN_CALLS)

        # ── DEPT VALIDATION: frontend department vs question department ───
        # Step 3 — Judge / Compare predicted department with actual
        if bot_dept.lower() == department.lower():
            # Both are same -> Answer it
            passed, reason = judge_answer(original_a, bot_answer)
        else:
            # Different -> AI should give mismatch answer telling user to switch to correct department
            expected_mismatch = f"switch to the {bot_dept} department"
            if bot_response.get("mode") == "mismatch" and (expected_mismatch.lower() in bot_answer.lower() or "please switch to" in bot_answer.lower()):
                passed = True
                reason = f"Detected department mismatch redirection to {bot_dept}."
            else:
                passed = False
                reason = f"Expected mismatch redirection to {bot_dept}, but got: {bot_answer}"
        time.sleep(DELAY_BETWEEN_CALLS)

        results.append({
            "faq_id": faq_id,
            "department": department,
            "original_q": original_q,
            "original_a": original_a,
            "rephrased_q": rephrased_q,
            "bot_answer": bot_answer,
            "bot_dept": bot_dept,
            "passed": passed,
            "reason": reason,
        })

        status = "PASS" if passed else "FAIL"
        print(f"  Verdict    : {status}")
        print(f"  Reason     : {reason}")
        print()

    elapsed = time.time() - start_time

    # ── Run Grounding and Memory Isolation Tests ─────────────────────────
    print("=" * 72)
    print("  Running Grounding & Memory Isolation Tests")
    print("=" * 72)
    print()
    
    grounding_passed = test_grounding_strictness()
    time.sleep(DELAY_BETWEEN_CALLS)
    memory_passed = test_memory_isolation()
    time.sleep(DELAY_BETWEEN_CALLS)
    out_of_scope_passed = test_out_of_scope_and_formatting()
    
    results.append({
        "faq_id": "GROUNDING",
        "department": "HR",
        "original_q": "Vague grounding test",
        "original_a": "No generic department/responsibilities hallucination",
        "rephrased_q": "do you know about hr of pel",
        "bot_answer": "",
        "bot_dept": "HR",
        "passed": grounding_passed,
        "reason": "Passed grounding checks" if grounding_passed else "Failed grounding checks",
    })
    
    results.append({
        "faq_id": "MEMORY",
        "department": "HR",
        "original_q": "Memory isolation test",
        "original_a": "No turn 1 context leak in turn 3",
        "rephrased_q": "do you know about hr of pel",
        "bot_answer": "",
        "bot_dept": "HR",
        "passed": memory_passed,
        "reason": "Passed memory isolation checks" if memory_passed else "Failed memory isolation checks",
    })

    results.append({
        "faq_id": "OUT_OF_SCOPE",
        "department": "Unknown",
        "original_q": "Out-of-scope and formatting test",
        "original_a": "No department guess (Unknown) and no duplicate word ending",
        "rephrased_q": "who is the CEO of PEL",
        "bot_answer": "",
        "bot_dept": "Unknown",
        "passed": out_of_scope_passed,
        "reason": "Passed out-of-scope and formatting checks" if out_of_scope_passed else "Failed out-of-scope or formatting checks",
    })
    print()

    total = len(results)

    # ── Summary ───────────────────────────────────────────────────────────
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total - passed_count
    score_pct = (100 * passed_count / total) if total > 0 else 0

    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print()
    print(f"  {'FAQ ID':<12} {'Department':<20} {'Result':<10}")
    print(f"  {'-'*12} {'-'*20} {'-'*10}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['faq_id']:<12} {r['department']:<20} {status:<10}")

    # ── Per-Department Breakdown ──────────────────────────────────────────
    dept_stats: dict[str, dict] = {}
    for r in results:
        dept = r["department"]
        if dept not in dept_stats:
            dept_stats[dept] = {"total": 0, "passed": 0}
        dept_stats[dept]["total"] += 1
        if r["passed"]:
            dept_stats[dept]["passed"] += 1

    print()
    print(f"  {'Department':<20} {'Pass':<6} {'Total':<6} {'Rate':<8}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8}")
    for dept, stats in sorted(dept_stats.items()):
        rate = (100 * stats["passed"] / stats["total"]) if stats["total"] > 0 else 0
        print(f"  {dept:<20} {stats['passed']:<6} {stats['total']:<6} {rate:.0f}%")

    print()
    print(f"  Total:    {total}")
    print(f"  Passed:   {passed_count}")
    print(f"  Failed:   {failed_count}")
    print(f"  Score:    {passed_count}/{total} ({score_pct:.0f}%)")
    print(f"  Duration: {elapsed:.1f}s")
    print()

    if failed_count > 0:
        print("  --- Failed Details ---")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['faq_id']}: {r['reason']}")
                print(f"    Original Q : {r['original_q']}")
                print(f"    Rephrased Q: {r['rephrased_q']}")
                if r["bot_answer"]:
                    print(f"    Bot Answer : {r['bot_answer'][:150]}...")
                print()

    # ── Save JSON Report ──────────────────────────────────────────────────
    REPORT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"eval_rephrase_{timestamp}.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "api_url": API_URL,
        "faq_pool_size": len(faqs),
        "tested": total,
        "passed": passed_count,
        "failed": failed_count,
        "score_pct": round(score_pct, 1),
        "duration_seconds": round(elapsed, 1),
        "seed": args.seed,
        "department_breakdown": dept_stats,
        "results": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  Report saved to: {report_path}")
    print()

    if failed_count > 0:
        print("  [!] Some tests failed. Review output above.")
        sys.exit(1)
    else:
        print("  All rephrase tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
