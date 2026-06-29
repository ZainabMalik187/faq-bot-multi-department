import sys
import uuid
import time
import requests

API_URL = "http://127.0.0.1:8000/chat"
DELAY_BETWEEN_CALLS = 2

def test_conversational_memory() -> bool:
    """
    Tests name recall across turns.
    Sends name first, then asks for the name.
    """
    print("Running test_conversational_memory...")
    session_id = f"test_memory_{uuid.uuid4()}"
    
    # Turn 1: Tell the bot the name
    payload1 = {
        "department": "HR",
        "text": "hi, my name is Ahmed",
        "session_id": session_id
    }
    try:
        resp1 = requests.post(API_URL, json=payload1, timeout=60)
        resp1.raise_for_status()
        data1 = resp1.json()
        print(f"  Turn 1 Query: {payload1['text']}")
        print(f"  Turn 1 Ans  : {data1.get('answer')}")
    except Exception as e:
        print(f"  [FAIL] Turn 1 request failed: {e}")
        return False
        
    time.sleep(DELAY_BETWEEN_CALLS)
    
    # Turn 2: Ask the bot the name
    payload2 = {
        "department": "HR",
        "text": "what is my name?",
        "session_id": session_id
    }
    try:
        resp2 = requests.post(API_URL, json=payload2, timeout=60)
        resp2.raise_for_status()
        data2 = resp2.json()
        ans2 = data2.get("answer", "")
        print(f"  Turn 2 Query: {payload2['text']}")
        print(f"  Turn 2 Ans  : {ans2}")
        
        if "ahmed" not in ans2.lower():
            print(f"  [FAIL] Bot failed to recall the name 'Ahmed' from conversation history.")
            return False
            
        print("  [PASS] Conversational memory recall test passed.")
        return True
    except Exception as e:
        print(f"  [FAIL] Turn 2 request failed: {e}")
        return False


def test_mode_switching() -> bool:
    """
    Tests switching from chat to FAQ and back.
    - Start chat (conversational)
    - Query FAQ (faq)
    - Ask follow-up/conversational (conversational)
    """
    print("Running test_mode_switching...")
    session_id = f"test_switching_{uuid.uuid4()}"
    
    # Turn 1: General greeting (Conversational)
    payload1 = {
        "department": "HR",
        "text": "hello, I am looking for some help",
        "session_id": session_id
    }
    try:
        resp1 = requests.post(API_URL, json=payload1, timeout=60)
        resp1.raise_for_status()
        data1 = resp1.json()
        print(f"  Turn 1 Query: {payload1['text']}")
        print(f"  Turn 1 Ans  : {data1.get('answer')}")
        print(f"  Turn 1 Mode : {data1.get('mode')}")
        
        if data1.get("mode") != "conversational":
            print(f"  [FAIL] Expected mode to be 'conversational', got '{data1.get('mode')}'")
            return False
    except Exception as e:
        print(f"  [FAIL] Turn 1 request failed: {e}")
        return False
        
    time.sleep(DELAY_BETWEEN_CALLS)
    
    # Turn 2: Specific FAQ policy query (FAQ)
    payload2 = {
        "department": "HR",
        "text": "how do I apply for annual leave",
        "session_id": session_id
    }
    try:
        resp2 = requests.post(API_URL, json=payload2, timeout=60)
        resp2.raise_for_status()
        data2 = resp2.json()
        print(f"  Turn 2 Query: {payload2['text']}")
        print(f"  Turn 2 Ans  : {data2.get('answer')}")
        print(f"  Turn 2 Mode : {data2.get('mode')}")
        print(f"  Turn 2 Dept : {data2.get('department')}")
        
        if data2.get("mode") != "faq":
            print(f"  [FAIL] Expected mode to be 'faq', got '{data2.get('mode')}'")
            return False
            
        if data2.get("department") != "HR":
            print(f"  [FAIL] Expected department to be 'HR', got '{data2.get('department')}'")
            return False
    except Exception as e:
        print(f"  [FAIL] Turn 2 request failed: {e}")
        return False
        
    time.sleep(DELAY_BETWEEN_CALLS)
    
    # Turn 3: Conversational follow-up (Conversational)
    payload3 = {
        "department": "HR",
        "text": "thanks, that is clear. How are you doing today?",
        "session_id": session_id
    }
    try:
        resp3 = requests.post(API_URL, json=payload3, timeout=60)
        resp3.raise_for_status()
        data3 = resp3.json()
        print(f"  Turn 3 Query: {payload3['text']}")
        print(f"  Turn 3 Ans  : {data3.get('answer')}")
        print(f"  Turn 3 Mode : {data3.get('mode')}")
        
        if data3.get("mode") != "conversational":
            print(f"  [FAIL] Expected mode to be 'conversational', got '{data3.get('mode')}'")
            return False
            
        print("  [PASS] Mode switching test passed.")
        return True
    except Exception as e:
        print(f"  [FAIL] Turn 3 request failed: {e}")
        return False

# ── SCOPE FIX: out-of-scope rejection ────────────────────────────
def test_outofscope_rejection() -> bool:
    """
    Tests that the bot rejects general knowledge queries as out-of-scope.
    """
    print("Running test_outofscope_rejection...")
    OUT_OF_SCOPE_MESSAGE = (
        "I'm only able to help with PEL-related questions. "
        "Please ask me about HR, IT, Sales, Finance, or "
        "Customer Support topics."
    )
    
    test_queries = [
        "what is AI",
        "what is PostgreSQL",
        "how to write code",
        "who is the president",
        "explain machine learning"
    ]
    
    for query in test_queries:
        payload = {
            "department": "HR",
            "text": query,
            "session_id": f"test_scope_{uuid.uuid4()}"
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            ans = data.get("answer", "")
            mode = data.get("mode", "")
            dept = data.get("department", "")
            print(f"  Query: {query}")
            print(f"  Ans  : {ans}")
            print(f"  Mode : {mode}")
            print(f"  Dept : {dept}")
            
            if ans != OUT_OF_SCOPE_MESSAGE:
                print(f"  [FAIL] Expected answer to be '{OUT_OF_SCOPE_MESSAGE}', got '{ans}'")
                return False
            if mode != "outofscope":
                print(f"  [FAIL] Expected mode to be 'outofscope', got '{mode}'")
                return False
            if dept != "General":
                print(f"  [FAIL] Expected department to be 'General', got '{dept}'")
                return False
        except Exception as e:
            print(f"  [FAIL] Request failed for query '{query}': {e}")
            return False
            
    print("  [PASS] Out-of-scope rejection test passed.")
    return True


# ── DEPT VALIDATION: frontend department vs question department ───
def test_department_mismatch() -> bool:
    """
    Tests that the bot detects department mismatch between
    frontend-selected department and actual question department.
    """
    print("Running test_department_mismatch...")

    # Mismatch cases: frontend dept != actual question dept
    mismatch_cases = [
        {"dept": "HR",      "query": "how do I track my order"},
        {"dept": "IT",      "query": "how do I apply for leave"},
        {"dept": "Finance", "query": "my laptop is not working"},
        {"dept": "Sales",   "query": "what is the leave policy"},
    ]

    for case in mismatch_cases:
        payload = {
            "department": case["dept"],
            "text": case["query"],
            "session_id": f"test_mismatch_{uuid.uuid4()}"
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            ans = data.get("answer", "")
            mode = data.get("mode", "")
            dept = data.get("department", "")
            print(f"  Query: {case['query']} (frontend: {case['dept']})")
            print(f"  Ans  : {ans}")
            print(f"  Mode : {mode}")
            print(f"  Dept : {dept}")

            if mode != "mismatch":
                print(f"  [FAIL] Expected mode to be 'mismatch', got '{mode}'")
                return False
            if "Please switch to" not in ans:
                print(f"  [FAIL] Expected 'Please switch to' in answer")
                return False
            if dept == case["dept"]:
                print(f"  [FAIL] Returned department should differ from frontend dept '{case['dept']}'")
                return False
        except Exception as e:
            print(f"  [FAIL] Request failed for query '{case['query']}': {e}")
            return False
        time.sleep(DELAY_BETWEEN_CALLS)

    # Matching cases: frontend dept == actual question dept
    matching_cases = [
        {"dept": "HR",               "query": "how do I apply for leave"},
        {"dept": "Customer Support", "query": "how do I track my order"},
        {"dept": "IT",               "query": "how do I reset my password"},
    ]

    for case in matching_cases:
        payload = {
            "department": case["dept"],
            "text": case["query"],
            "session_id": f"test_match_{uuid.uuid4()}"
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            ans = data.get("answer", "")
            mode = data.get("mode", "")
            dept = data.get("department", "")
            print(f"  Query: {case['query']} (frontend: {case['dept']})")
            print(f"  Ans  : {ans}")
            print(f"  Mode : {mode}")
            print(f"  Dept : {dept}")

            if mode != "faq":
                print(f"  [FAIL] Expected mode to be 'faq', got '{mode}'")
                return False
            if "Please switch to" in ans:
                print(f"  [FAIL] Answer should NOT contain 'Please switch to' for matching dept")
                return False
        except Exception as e:
            print(f"  [FAIL] Request failed for query '{case['query']}': {e}")
            return False
        time.sleep(DELAY_BETWEEN_CALLS)

    print("  [PASS] Department mismatch test passed.")
    return True


def main():
    global API_URL
    import argparse
    parser = argparse.ArgumentParser(description="Run Conversational and Mode Switching Tests")
    parser.add_argument("--port", type=int, default=8000, help="API Port")
    args = parser.parse_args()

    API_URL = f"http://127.0.0.1:{args.port}/chat"

    print("=" * 60)
    print(f"  Running Conversational and Mode Switching Tests (Port {args.port})")
    print("=" * 60)
    
    mem_pass = test_conversational_memory()
    print()
    switch_pass = test_mode_switching()
    print()
    scope_pass = test_outofscope_rejection()
    print()
    # ── DEPT VALIDATION: frontend department vs question department ───
    dept_pass = test_department_mismatch()
    print()
    
    if mem_pass and switch_pass and scope_pass and dept_pass:
        print("All evaluate.py tests passed successfully!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
