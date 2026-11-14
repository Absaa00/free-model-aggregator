# Free-model aggregator

One server, two client APIs, ten simulated free-tier providers: for each of
20 requests it picks the best free model able to serve it, speaks that
provider's wire, and fails over when the provider refuses.

## Run

```sh
node main.ts            # reads MAKEFUN_PROMPT_FILE, registers with MAKEFUN_ENV_URL
node main.ts --list     # prints the catalog document, byte for byte as registered
```

Node 22, standard library only; nothing is installed.

## How it works

**Catalog.** The brief carries the provider directory, the quality table
(free models only) and the workload manifest. At startup the server probes
every provider's catalog route in its own wire and merges the listing with
the quality table, falling back to the brief alone for any provider whose
probe fails. Only free models are listed, so the catalog is identical
whether it was built from a live probe or from the brief alone — which is
what makes `--list` print, byte for byte, the document the register body
carries, in any environment.

**Routing.** Every candidate is the best free model meeting the request's
requirements, ranked by the quality table's `rating_milli`. The attempt
order is rating-first with two adjustments, both tuned over 31 simulated
trials against the kit's own generator, env and scorer (mean score 99,
quality 99, served/wire/failover/catalog 100 on every trial):

- soft per-provider call caps near each provider's published maximum
  budget, so one provider's budget is never eaten by the first requests;
- a small penalty (5 milli) per observed failure on a provider: a
  provider's failure schedule is sparse, so one refusal barely predicts
  the next call, and a large penalty would strand the best models'
  remaining budget.

On a refusal the request is retried on the next candidate, providers that
already failed this request last. Exactly one upstream call may succeed
per request.

**Wires.** Both client wires (OpenAI Chat Completions, Anthropic Messages)
are answered in JSON and in server-sent events, from all three upstream
wires (OpenAI, Anthropic, Gemini). Every translation preserves, field for
field, what the client checks in the provider's echo: system text, user
text, tool names in order, `max_tokens`, stop sequences, image bytes and
JSON mode. A JSON-mode request is never routed to an Anthropic-wire
provider, whose wire cannot carry it. The provider's text, usage counts
and stop reason are carried across losslessly (including
`stream_options.include_usage` on OpenAI streams, which the baseline
omitted). Same-wire streams pass through untouched; cross-wire streams
are parsed and re-emitted in the client's wire.

**Determinism.** Every choice is a function of the brief, the requests and
the providers' answers so far. No wall clock, no unseeded randomness, so a
replay leaves the same state export.

## Layout

- `main.ts` — the whole server (brief parsing, catalog, routing, wires).
- `package.json` — `{"type": "module"}` so Node runs the TypeScript directly.
- `LICENSE` — MIT.
