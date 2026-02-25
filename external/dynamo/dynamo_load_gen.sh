#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dynamo KV Cache Load Generator
#
# Sends randomized chat completion requests indefinitely to fill the KV cache.
# Each request contains a ~7000-token intelligence report with UUID-randomized
# entity names so every prompt is unique (no prefix-cache hits).
#
# Usage:
#   bash dynamo_load_gen.sh                          # defaults: 1 req/s, 4 concurrent
#   bash dynamo_load_gen.sh -f 5 -c 10              # 5 req/s, 10 concurrent
#   bash dynamo_load_gen.sh --pin                    # enable cache_control pinning
#   bash dynamo_load_gen.sh -f 2 -c 8 --pin --stream
#
#   # Run in background:
#   nohup bash dynamo_load_gen.sh -f 5 -c 10 &> /tmp/load_gen.log &
#
#   # Stop:
#   kill %1   OR   kill $(cat /tmp/dynamo-load-gen.pid)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

if ! command -v jq &>/dev/null; then
    echo "ERROR: jq is required but not found. Install with: apt install jq" >&2
    exit 1
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
FREQUENCY=1
CONCURRENCY=4
ENDPOINT=""
MODEL=""
MAX_TOKENS=256
STREAM=false
PIN_CACHE=false
PIN_TTL="5m"
VERBOSE=false

usage() {
    cat <<'EOF'
Usage: bash dynamo_load_gen.sh [OPTIONS]

Options:
  -f, --frequency NUM     Requests per second (default: 1)
  -c, --concurrency NUM   Max concurrent requests (default: 4)
  -e, --endpoint URL      API endpoint (default: http://localhost:$DYNAMO_HTTP_PORT/v1/chat/completions)
  -m, --model NAME        Model name (default: basename of $DYNAMO_MODEL_DIR)
  -t, --max-tokens NUM    Max tokens per response (default: 256)
  -s, --stream            Enable streaming responses (holds connections open longer)
  -p, --pin               Add nvext.cache_control for cache pinning
      --pin-ttl DURATION  Cache pin TTL (default: 5m, requires --pin)
  -v, --verbose           Print full response bodies
  -h, --help              Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--frequency)    FREQUENCY="$2"; shift 2 ;;
        -c|--concurrency)  CONCURRENCY="$2"; shift 2 ;;
        -e|--endpoint)     ENDPOINT="$2"; shift 2 ;;
        -m|--model)        MODEL="$2"; shift 2 ;;
        -t|--max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
        -s|--stream)       STREAM=true; shift ;;
        -p|--pin)          PIN_CACHE=true; shift ;;
        --pin-ttl)         PIN_TTL="$2"; shift 2 ;;
        -v|--verbose)      VERBOSE=true; shift ;;
        -h|--help)         usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

HTTP_PORT="${DYNAMO_HTTP_PORT:-8099}"
MODEL_DIR="${DYNAMO_MODEL_DIR:-$HOME/models/Llama-3.3-70B-Instruct}"

[ -z "$ENDPOINT" ] && ENDPOINT="http://localhost:${HTTP_PORT}/v1/chat/completions"
[ -z "$MODEL" ] && MODEL="$(basename "$MODEL_DIR")"

INTERVAL=$(awk "BEGIN {printf \"%.6f\", 1.0 / $FREQUENCY}")

PIDFILE="/tmp/dynamo-load-gen.pid"
echo $$ > "$PIDFILE"

# ── Semaphore (FIFO-based, limits concurrent curl jobs) ──────────────────────
SEM_FIFO=$(mktemp -u /tmp/dynamo-load-gen-sem.XXXXXX)
mkfifo "$SEM_FIFO"
exec 3<>"$SEM_FIFO"
rm -f "$SEM_FIFO"

for ((i=0; i<CONCURRENCY; i++)); do
    echo >&3
done

# ── Counters (shared via temp files for subshell visibility) ─────────────────
COUNT_FILE=$(mktemp /tmp/dynamo-load-gen-count.XXXXXX)
echo 0 > "$COUNT_FILE"
ERR_FILE=$(mktemp /tmp/dynamo-load-gen-err.XXXXXX)
echo 0 > "$ERR_FILE"

# ── Cleanup ──────────────────────────────────────────────────────────────────
SHUTTING_DOWN=false

cleanup() {
    $SHUTTING_DOWN && return
    SHUTTING_DOWN=true
    echo ""
    echo "[load-gen] Shutting down..."
    local pids
    pids=$(jobs -p 2>/dev/null)
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null || true
        sleep 2
        kill -9 $pids 2>/dev/null || true
    fi
    wait 2>/dev/null
    local total err
    total=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
    err=$(cat "$ERR_FILE" 2>/dev/null || echo 0)
    echo "[load-gen] Done. Sent $total requests ($err errors)."
    rm -f "$COUNT_FILE" "$ERR_FILE" "$PIDFILE"
    exec 3>&-
}
trap cleanup EXIT INT TERM

# ── Banner ───────────────────────────────────────────────────────────────────
echo "========================================================="
echo "Dynamo KV Cache Load Generator"
echo "========================================================="
echo "Endpoint:     $ENDPOINT"
echo "Model:        $MODEL"
echo "Frequency:    $FREQUENCY req/s (interval=${INTERVAL}s)"
echo "Concurrency:  $CONCURRENCY"
echo "Max tokens:   $MAX_TOKENS"
echo "Stream:       $STREAM"
echo "Cache pin:    $PIN_CACHE (ttl=$PIN_TTL)"
echo "Prompt size:  ~7000 tokens (25-section intelligence report)"
echo "PID file:     $PIDFILE"
echo "========================================================="
echo ""
echo "[load-gen] Running... Ctrl+C or 'kill \$(cat $PIDFILE)' to stop."
echo ""

# ── Prompt builder (~7000 tokens, fully unique from token 0) ─────────────────
uuid() { cat /proc/sys/kernel/random/uuid; }

# Pre-generate a block of random hex to use as cheap random padding.
# Each call to rand_pad returns a unique substring.
RAND_PAD_OFFSET=0
rand_words() {
    local count=${1:-20}
    local words=()
    local i
    for ((i=0; i<count; i++)); do
        words+=("$(uuid)")
    done
    echo "${words[*]}"
}

build_section() {
    local section_type=$1 u1=$2 u2=$3 u3=$4 u4=$5

    case $section_type in
    field)
        echo "Field operative $u1, currently deployed under non-official cover in Sector $u2, "\
"has submitted an urgent assessment regarding the organization designated $u3. Over the "\
"past seventy-two hour surveillance window, the operative observed a marked increase in "\
"logistics activity centered around warehouse complex $u4. Personnel arriving at the "\
"facility were observed carrying documentation bearing official seals from the provincial "\
"trade authority. The operative notes that vehicle traffic has increased approximately "\
"threefold during nighttime hours, with most arrivals occurring between zero-one-hundred "\
"and zero-four-hundred local time. Initial signals intelligence suggests the organization "\
"is coordinating with at least two previously unidentified external entities. Supply "\
"manifests recovered from a discarded container indicate shipments of industrial equipment "\
"inconsistent with the facility registered commercial purpose. Local informant networks "\
"corroborate the increased activity and report unfamiliar personnel with technical "\
"equipment have been observed conducting surveys of adjacent properties. The operative "\
"recommends escalating surveillance priority and deploying additional technical collection "\
"assets to the area surrounding complex $u4. Risk assessment: moderate. Confidence level: "\
"high based on direct observation and corroborating signals data."
        ;;
    installation)
        echo "Automated monitoring station $u1, positioned at the perimeter of restricted zone $u2, "\
"has detected anomalous electromagnetic signatures consistent with the activation of "\
"previously dormant infrastructure. Analysis by technical team $u3 indicates the signatures "\
"originate from a subterranean facility approximately three hundred meters below ground "\
"level. The emission pattern matches historical profiles associated with advanced "\
"manufacturing operations. Thermal imaging from overhead assets confirms elevated heat "\
"signatures in a radial pattern extending two hundred meters from the suspected facility "\
"entrance at coordinates designated $u4. Ground-penetrating radar surveys conducted during "\
"the previous quarter had identified subsurface structures in this location but they were "\
"assessed as inactive at that time. The current activity level suggests a rapid mobilization "\
"of resources and personnel. Power consumption data obtained from regional utility "\
"monitoring shows a sustained forty percent increase in the electrical draw over the past "\
"nine days. Seismic sensors have also recorded low-frequency vibrations consistent with "\
"heavy machinery operation. Technical assessment recommends immediate deployment of "\
"specialized detection equipment to characterize the operational purpose of the facility "\
"beneath zone $u2."
        ;;
    financial)
        echo "Financial intelligence unit $u1 has completed a comprehensive analysis of transaction "\
"flows associated with accounts linked to entity $u2. The investigation reveals a complex "\
"network of shell organizations spanning fourteen jurisdictions, with the primary financial "\
"conduit operating through institution $u3. Transaction volumes have increased by "\
"approximately sixty percent over the preceding quarter, with individual transfers carefully "\
"structured to remain below regulatory reporting thresholds. The most significant finding "\
"involves a series of payments totaling substantial sums directed to procurement agent $u4, "\
"who has been previously identified as a facilitator for restricted technology acquisitions. "\
"Pattern analysis suggests the financial network employs a layering technique involving at "\
"least four intermediate accounts before funds reach their final destination. Currency "\
"conversion records indicate a preference for transactions denominated in multiple "\
"currencies to further obscure the money trail. Forensic accounting has identified "\
"discrepancies in reported revenues for three front companies that exceed normal variance "\
"by an order of magnitude. The financial intelligence unit assesses with high confidence "\
"that this network is actively facilitating the procurement of dual-use technologies and "\
"recommends coordination with partner agencies for joint disruption operations targeting "\
"the accounts held at institution $u3."
        ;;
    signals)
        echo "Signals intelligence collection platform $u1 has intercepted a series of encrypted "\
"communications between node $u2 and relay station $u3 over the past fourteen-day "\
"collection cycle. Cryptanalysis team assessment indicates the communications employ a "\
"modified encryption protocol not previously catalogued in existing databases. Despite "\
"the enhanced security measures, metadata analysis reveals a consistent communication "\
"schedule with transmissions occurring at six-hour intervals beginning at midnight "\
"coordinated universal time. Geolocation of the transmitting station places it within "\
"the administrative boundaries of district $u4, approximately twelve kilometers from the "\
"nearest known facility of interest. Voice pattern analysis of brief unencrypted preambles "\
"detected in three transmissions matches profiles of two individuals currently on the "\
"priority watch list. The volume of communications traffic has doubled compared to the "\
"previous collection period, suggesting an acceleration in operational planning. Network "\
"analysis indicates the relay station $u3 serves as a hub connecting at least seven "\
"additional nodes whose locations remain under investigation. Recommendations include "\
"deploying additional collection assets to achieve full-spectrum coverage and coordinating "\
"with partner agencies for potential decryption support of the novel protocol observed in "\
"district $u4."
        ;;
    strategic)
        echo "Strategic assessment division $u1 has completed its quarterly evaluation of developments "\
"in operational theater $u2. The assessment incorporates reporting from seventeen distinct "\
"collection sources and addresses the evolving capabilities and intentions of organization "\
"$u3. Key findings indicate a shift in the operational posture from defensive consolidation "\
"to active expansion into previously uncontested areas. Infrastructure development in zone "\
"$u4 has accelerated significantly, with construction of permanent facilities observed at "\
"three new locations. Human intelligence sources report increased recruitment activity "\
"targeting technically skilled individuals, particularly those with expertise in advanced "\
"manufacturing and systems engineering. The leadership structure appears to have undergone "\
"recent reorganization, with several previously peripheral figures assuming more prominent "\
"roles. Open source analysis of public communications reveals increasingly assertive "\
"messaging that contrasts with the more conciliatory tone observed in prior assessment "\
"periods. Economic indicators suggest the organization has secured new funding channels "\
"that are not yet fully characterized by financial intelligence. The strategic assessment "\
"division recommends elevating the threat assessment level for theater $u2 and initiating "\
"contingency planning for potential disruption scenarios within the next sixty to ninety "\
"days involving organization $u3."
        ;;
    esac
}

build_prompt() {
    local section_types=(field installation financial signals strategic)
    local num_sections=25

    # Shuffle section order using Fisher-Yates on the type sequence
    local order=()
    local i
    for ((i=0; i<num_sections; i++)); do
        order+=($i)
    done
    for ((i=num_sections-1; i>0; i--)); do
        local j=$((RANDOM % (i + 1)))
        local tmp=${order[$i]}
        order[$i]=${order[$j]}
        order[$j]=$tmp
    done

    # Start with a fully unique preamble -- UUIDs at token 0 prevent prefix caching
    local doc_id=$(uuid)
    local class_id=$(uuid)
    local dir_id=$(uuid)
    local session_id=$(uuid)
    local analyst_id=$(uuid)

    local prompt=""
    prompt+="DOCUMENT $doc_id SESSION $session_id ANALYST $analyst_id\n"
    prompt+="CLASSIFICATION $class_id DIRECTORATE $dir_id\n"
    prompt+="RANDOM SEED $(rand_words 6)\n\n"
    prompt+="You are a senior intelligence analyst reviewing a classified compilation of field reports. "
    prompt+="Read the entire document below and then write exactly one paragraph summarizing the three "
    prompt+="most critical developments and their strategic implications.\n\n"

    local section_num=0
    for i in "${order[@]}"; do
        local stype=${section_types[$((i % ${#section_types[@]}))]}
        local u1=$(uuid) u2=$(uuid) u3=$(uuid) u4=$(uuid)
        local extra1=$(uuid) extra2=$(uuid)
        section_num=$((section_num + 1))

        prompt+="--- SECTION $section_num TYPE ${stype^^} REF $extra1 CROSS-REF $extra2 ---\n"
        prompt+="$(build_section "$stype" "$u1" "$u2" "$u3" "$u4")\n\n"
    done

    prompt+="=== END OF COMPILATION $doc_id ===\n\n"
    prompt+="Based on all twenty-five reports above, write exactly one paragraph identifying the three "
    prompt+="most critical developments across all reporting disciplines and assess their combined "
    prompt+="strategic implications for the next ninety-day planning cycle."

    printf '%s' "$prompt"
}

# ── Request sender ───────────────────────────────────────────────────────────
send_request() {
    local req_num=$1
    local req_id
    req_id=$(uuid)

    local prompt
    prompt=$(build_prompt)

    local body
    body=$(jq -n \
        --arg model "$MODEL" \
        --arg content "$prompt" \
        --argjson max_tokens "$MAX_TOKENS" \
        --argjson stream "$STREAM" \
        '{model: $model, messages: [{role: "user", content: $content}], max_tokens: $max_tokens, temperature: 0.9, stream: $stream}')

    if [ "$PIN_CACHE" = true ]; then
        body=$(echo "$body" | jq --arg ttl "$PIN_TTL" \
            '.nvext = {cache_control: {type: "ephemeral", ttl: $ttl}}')
    fi

    local response
    response=$(curl -s -w "\n%{http_code} %{time_total}" \
        --max-time 300 \
        "$ENDPOINT" \
        -H 'Content-Type: application/json' \
        -d "$body" 2>&1)

    local status_line http_code time_total
    status_line=$(echo "$response" | tail -1)
    http_code=$(echo "$status_line" | awk '{print $1}')
    time_total=$(echo "$status_line" | awk '{print $2}')

    local ts
    ts=$(date '+%H:%M:%S')

    if [[ "$http_code" == 200 ]]; then
        echo "[$ts] #$req_num  HTTP $http_code  ${time_total}s  id=$req_id"
    else
        echo "[$ts] #$req_num  HTTP $http_code  ${time_total}s  ERROR  id=$req_id" >&2
        if [ "$VERBOSE" = true ]; then
            echo "$response" | head -n -1 >&2
        fi
        local cur
        cur=$(cat "$ERR_FILE"); echo $((cur + 1)) > "$ERR_FILE"
    fi

    local cur
    cur=$(cat "$COUNT_FILE"); echo $((cur + 1)) > "$COUNT_FILE"

    echo >&3
}

# ── Main loop ────────────────────────────────────────────────────────────────
REQ_NUM=0
while ! $SHUTTING_DOWN; do
    if read -t 1 -u 3; then
        REQ_NUM=$((REQ_NUM + 1))
        send_request "$REQ_NUM" &
        sleep "$INTERVAL"
    fi
done
