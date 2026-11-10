# NOTES: how the submission works

**Submission folder:** `submission/` (entry point `submission/solution.py`, `build()`).
**Data:** the supplied website snapshots are **not** in this repository (they are not ours to redistribute). After cloning, copy the kit's `snapshots_zip/` folder (the five `.zip` files) into the repo root. The scorer unpacks them on its first run.
**Run:** `python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl`
**API keys:** none. **Paid APIs:** none. **Packages:** Python standard library only (`requirements.txt` is empty on purpose).
Everything is deterministic and runs offline. Ingesting all five snapshots takes about 15 seconds, and a typical answer takes 0.1 to 0.6 seconds.

## Approach

The system keeps **every snapshot whole** instead of merging them. Each page is cleaned: crawler mojibake, nav/footer lines that repeat across the site, tag rows, "related article" teasers and form labels are removed. Each page is then parsed into two things:

1. **Rate tables.** Every markdown table becomes rows with a product name (from the table title and section), qualifiers (term in months, balance or loan-amount tier, model-year range, jumbo/special/intro, Roth/Traditional, purchase/refinance, business), value columns (dividend rate, APY, APR, rate, minimums) and the "Rates are effective as of …" date under the table.
2. **Text chunks** for a BM25 index. Each chunk records the page type and, for news and blog posts, the publish date.

`answer(question, as_of)` picks the snapshot first: the last one taken on or before `as_of`, or the newest when there is no date. A date written in the question ("as of May 2026") counts as `as_of`. The question is then routed:

* **Rate questions** go to the table engine. It ranks tables by how well they name the product, then keeps only the rows that match the qualifiers asked for, so we quote the right tier and nothing else. If the member names a tier (a 60-month term, $15,000, a 2025 model year, jumbo), only that row is returned. If they ask generally ("mortgage rates"), the whole table is listed. Superlatives ("highest Share Savings APY") pick the extreme row.
* **Structured extractors** handle the routing number, branch addresses (the branch locator, or "located at …" in press releases), lobby and drive-up hours (schedules published in branch press releases; if branches differ, we list the schedules and ask which branch), membership eligibility, links ("what URL…"), and counts ("how many branches/members…").
* **Everything else** goes through BM25. The best 1–3 sentence window is picked from the top chunks, weighted by page authority and recency. FAQ questions keep their answer, people's quotes and bare buttons are penalised, and "when" questions prefer dates.

## How we decide what is current (the rules)

1. **The newest snapshot wins** unless `as_of` says otherwise. Each snapshot is the whole site as of that day.
2. **The official rate sheet beats everything else.** Authority order: `/rates` and `/business-rates` > product pages > landing pages / promos > PDFs > calculators > blog/news. Rates are quoted only from the rate sheets and product pages. Landing pages, calculators and PDFs are where stale numbers and expired promotions live (for example 5.34% on `/switch-and-earn` and the vehicle calculator vs 5.39% on `/rates`, or 6.037% on `/homebuying-guidance`), so they are used only to *detect* conflicts.
3. **When pages name the product equally well, the most authoritative one wins.** A stale page never wins because it matches one more word.
4. **Qualifiers are different facts, not conflicts.** Different terms, tiers, model years, rate vs APY, jumbo vs standard, personal vs business, Roth vs Traditional and purchase vs refinance are all kept apart, both in answers and in `conflicts()`.
5. **Newest publication wins for text facts.** Among press releases restating a fact (branch count, membership, "the fifth time Forbes recognised Dupaco"), the most recently published one wins. An article re-published under the same title supersedes the older ones.
6. **Be honest.** If the product, term or branch the member names isn't on the site (7/1 ARM, construction loan, 60-month jumbo certificate, a Mullan Avenue branch), or the best passage doesn't cover the question, we answer "I don't know… please contact Dupaco" with `confident=False` and no numbers.

## conflicts()

One entry per disputed fact. Rows are keyed by canonical product + term + tier + model-year rank + variant flags, so different tiers are never reported as conflicts.

* **Rate sheet changes over time:** the 10-month jumbo special went from 3.90% to 4.00%, 1st Rate Checking from 3.00% to 2.50%, plus auto, certificate and mortgage moves. Resolution: the newest official value.
* **Stale secondary pages in the newest snapshot that disagree with the rate sheet:** auto rates on `/switch-and-earn` and the vehicle calculator, mortgage rates on `/homebuying-guidance` and `/mortgage-refi`. Resolution: the rate-sheet value.
* **Inconsistencies inside the rate sheet:** the Roth vs Traditional 10-month jumbo IRA.
* **Text facts restated in press releases:** branch count (19/20/22/23/24 → 22, the newest) and Forbes recognitions (fourth → fifth).

Mirrored disputes (the same change on the business or IRA copy of a certificate) are reported once to keep precision high.

## Testing

* Public eval: `python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl` gives QA 0.946. The two misses are an item with an empty `must_include` (it cannot score) and a VTO count whose expected answer comes from the older of two annual articles.
* Reworded questions: `python tools/check_paraphrases.py` runs 100 checks: rewordings of the public questions, rate tiers, as-of dates and refusals, using the scorer's own matching (96/100 pass).
* Harness self-test site: `python harness/run_eval.py --submission submission --snapshots harness/selftest/snapshots --eval harness/selftest/eval.jsonl --conflicts-gold harness/selftest/conflicts_gold.json` gives 1.0, which shows the rules are not specific to one website.

## Known limitations

Free-text questions rely on lexical retrieval (BM25 plus rules), not a language model. Questions phrased very differently from the site's wording can retrieve a nearby but wrong sentence. The coverage check turns many of those into an honest "I don't know", but not all of them.
