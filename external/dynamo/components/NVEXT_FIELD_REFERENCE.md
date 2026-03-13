# nvext Field Reference

Fields injected into the OpenAI-compatible request body by NAT's `_DynamoTransport`.
All fields live under the top-level `nvext` key in the JSON request body.

---

## `nvext.agent_hints`

Routing hints consumed by Dynamo's built-in router/scheduler and the custom `processor.py`.

| Field | Description | Used in NAT Source/Components | Used in Dynamo Source/Components |
|---|---|---|---|
| `prefix_id` | Unique string identifying the KV cache prefix for a conversation. | Yes — `DynamoPrefixContext` generates it; `processor.py` routes on it. | No |
| `total_requests` | Expected number of LLM calls in this conversation. | Yes — `processor.py` computes `reuse_budget` from it. | No |
| `osl` | Expected output tokens (always raw integer in `agent_hints`). Config accepts `LOW`/`MEDIUM`/`HIGH` strings for backward compat (mapped to 128/512/2048). | Yes — `processor.py` uses it for router decode cost weighting. Validated for pass-through to Dynamo. | Yes — native `AgentHints.osl` read by Dynamo's built-in frontend. |
| `iat` | Expected inter-arrival time in milliseconds (always raw integer). Config accepts `LOW`/`MEDIUM`/`HIGH` strings for backward compat (mapped to 50/250/750). | Yes — `processor.py` uses it for router stickiness weighting. Also used client-side to compute `cache_control.ttl`. | No |
| `latency_sensitivity` | How latency-sensitive this request is (from `@latency_sensitive` decorator or prediction trie). Increase to prioritize a request. | Validated, passed through to Dynamo. Sets a context variable via `Context.push_latency_sensitivity()`. | Yes — native `AgentHints.latency_sensitivity` → Dynamo queue ordering. |
| `priority` | Engine scheduling priority (`nvext_max_sensitivity - latency_sensitivity`). Lower values = higher priority for vLLM; SGLang is configurable. | Validated, passed through to Dynamo. | Yes — native `AgentHints.priority` → engine queue, eviction, preemption. |

---

## `nvext.cache_control`

KV cache lifetime management. Injected as a sibling of `agent_hints` under `nvext`.
Only injected when `nvext_cache_pin_type` is set (not `None`).

| Field | Description | Used in NAT Source/Components | Used in Dynamo Source/Components |
|---|---|---|---|
| `type` | Cache pinning strategy. Only valid value is `"ephemeral"`. Required in JSON (no serde default on deserialization). | `dynamo_llm.py`: Injected client-side via `CachePinType.EPHEMERAL` (default). Configurable via `nvext_cache_pin_type` param; set to `null` to disable `cache_control` entirely. | `nvext.rs`: Deserialized into `CacheControlType` enum (single variant: `Ephemeral`). Presence of `cache_control` triggers `pin_prefix` after generation in the KV push router. Requires `--enable-cache-control` on the frontend. |
| `ttl` | Duration string for how long to pin the prefix in the KV cache. Optional; defaults to `"5m"` (300s) when omitted. | `dynamo_llm.py`: Computed client-side as `total_requests * iat` (ms), converted to seconds, formatted as `"<N>m"` (whole minutes) or `"<N>s"`. The `iat` field is not consumed by Dynamo — it is only used here for this TTL computation and by the custom Thompson Sampling `processor.py`. | `nvext.rs` `CacheControl::ttl_seconds()`: Only parses `"5m"` (300s) and `"1h"` (3600s). Any other value logs a warning and falls back to 300s. The parsed TTL is forwarded as `ttl_seconds` to `pin_prefix` on the worker via the `cache_control` service mesh endpoint (`cache_control.rs` / `handler_base.py`). |

### `nvext_cache_control_mode` (NAT config field)

Controls **when** `nvext.cache_control` is injected (not a wire-format field — only affects client-side behavior):

| Mode | Behavior |
|---|---|
| `always` (default) | Inject `cache_control` on every request. Refreshes TTL each turn. |
| `first_only` | Inject only on the first request per `prefix_id`. Pins the system prompt when first established in the KV cache; subsequent requests benefit from prefix matching without re-pinning the growing conversation context. |

This field is only relevant when `nvext_cache_pin_type` is set (i.e., `"ephemeral"`). When `nvext_cache_pin_type` is `null`, no `cache_control` is injected regardless of this mode.

---

## NAT Config → Wire Format Mapping

| NAT Config Field | Wire Format Location | Notes |
|---|---|---|
| `enable_nvext_hints` | *(gating only)* | When `false` (default), no `nvext` injection occurs. |
| `nvext_prefix_id_template` | *(unused by transport)* | Prefix IDs come from `DynamoPrefixContext` at request time. |
| `nvext_prefix_total_requests` | `nvext.agent_hints.total_requests` | |
| `nvext_prefix_osl` | `nvext.agent_hints.osl` | Always sent as raw integer. |
| `nvext_prefix_iat` | `nvext.agent_hints.iat` | Always sent as raw integer. Also used to compute `cache_control.ttl`. |
| `nvext_max_sensitivity` | `nvext.agent_hints.priority` | `priority = nvext_max_sensitivity - latency_sensitivity` |
| *(from Context)* | `nvext.agent_hints.latency_sensitivity` | Set by `@latency_sensitive` decorator or prediction trie. |
| *(from DynamoPrefixContext)* | `nvext.agent_hints.prefix_id` | Auto-generated per workflow run + call depth. |
| `nvext_cache_pin_type` | `nvext.cache_control.type` | `null` disables `cache_control` entirely. |
| `nvext_cache_control_mode` | *(gating only)* | Controls whether `cache_control` is injected on every request or just the first. |
| `nvext_prediction_trie_path` | *(overrides agent_hints values)* | When set, per-call predictions override `total_requests`, `osl`, `iat`, and optionally `latency_sensitivity`. |
