#!/usr/bin/env bash
# Test Remis's fact-extraction prompt against whatever model is running on
# the local llama_cpp.server.
#
# Usage: ./test_remi_extraction.sh [port]
#   defaults to 8081 (Remis's port) — pass 8080 to test core's model instead.

set -u

PORT="${1:-8080}"
URL="http://localhost:${PORT}/v1/chat/completions"

PROMPT_HEADER=$'Extract only facts the user explicitly stated that would still be worth knowing weeks from now — their name, job, relationships, preferences, or ongoing plans.\n\nSkip greetings, small talk, questions, and anything about the assistant itself. If nothing qualifies, return an empty list.\n\nAlways split distinct facts into separate entries in the list — never combine multiple facts into one sentence, even if they were stated together.\n\nExamples:\nInput: hi\nOutput: {"facts": []}\nInput: my name is Alex and I work as a nurse\nOutput: {"facts": ["User\'s name is Alex", "User works as a nurse"]}\nInput: I\'m K5031 and I go to UCL\nOutput: {"facts": ["User\'s name is K5031", "User attends UCL"]}\nInput: what\'s your name?\nOutput: {"facts": []}\n\n'

run_case() {
    local name="$1"
    local conversation="$2"
    local expect="$3"  # "empty" | "nonempty" | an integer = expected fact count

    local content="${PROMPT_HEADER}Conversation:
${conversation}

Return only valid JSON: {\"facts\": [...]}"

    # Pass content straight to python via stdin — no shell reinterpretation,
    # no printf, no risk of mangling embedded quotes/newlines.
    local payload
    payload=$(python3 -c "
import json, sys
content = sys.stdin.read()
print(json.dumps({
    'model': 'local-model',
    'messages': [{'role': 'user', 'content': content}],
    'temperature': 0.0,
    'max_tokens': 300
}))
" <<< "$content")

    local response
    response=$(curl -s "$URL" -H "Content-Type: application/json" -d "$payload")

    local facts_json
    facts_json=$(echo "$response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['message']['content'])
except Exception as e:
    print(f'PARSE_ERROR: {e}')
")

    local fact_count
    fact_count=$(echo "$facts_json" | python3 -c "
import json, sys, re
raw = sys.stdin.read()
match = re.search(r'\{.*\}', raw, re.DOTALL)
try:
    data = json.loads(match.group(0)) if match else {}
    print(len(data.get('facts', [])))
except Exception:
    print(-1)
")

    local status="?"
    case "$expect" in
        empty)
            [ "$fact_count" = "0" ] && status="PASS" || status="FAIL"
            ;;
        nonempty)
            [ "$fact_count" -gt "0" ] 2>/dev/null && status="PASS" || status="FAIL"
            ;;
        *)
            # numeric: expect an exact fact count (e.g. "2" for a split case)
            [ "$fact_count" = "$expect" ] && status="PASS" || status="FAIL"
            ;;
    esac

    printf "[%s] %-30s (facts=%-3s expect=%-9s) -> %s\n" "$status" "$name" "$fact_count" "$expect" "$facts_json"
}

echo "=== Testing Remis's extraction prompt (port $PORT) ==="
echo

run_case "small talk"              $'user: hi\nassistant: Hello! How can I help?'                                empty
run_case "meta-conversation"       $'user: what\'s your name?\nassistant: I\'m K.'                               empty
run_case "real fact (name)"        $'user: my name is K5031\nassistant: Nice to meet you, K5031.'                1
run_case "multi-fact (job+pet)"    $'user: I\'m a teacher and I have a dog named Rex\nassistant: That\'s great!'  2
run_case "multi-fact (name+uni)"   $'user: I\'m K5031 and I go to UCL\nassistant: Nice to meet you.'             2
run_case "small talk (repeat 1)"   $'user: hi\nassistant: Hello!'                                                empty
run_case "small talk (repeat 2)"   $'user: hi\nassistant: Hello!'                                                empty
run_case "small talk (repeat 3)"   $'user: hi\nassistant: Hello!'                                                empty

# Third-party relationship facts — this exact phrasing failed to extract
# anything in the full pipeline test (case 6), isolating it here to see
# whether it's an extraction-prompt gap or something else.
run_case "relationship (friend name)"   $'user: my friend\'s name is Alex\nassistant: Nice to meet them.'          1
run_case "relationship (sibling)"       $'user: my sister lives in Manchester\nassistant: That\'s nice.'           1
run_case "relationship (partner job)"   $'user: my wife works as a doctor\nassistant: That\'s a great career.'     1

echo
echo "=== Done ==="