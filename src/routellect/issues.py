from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IssueSpec:
    issue_id: str
    title: str
    description: str
    labels: list[str] | None = None


ISSUE_437 = IssueSpec(
    issue_id="437",
    title="Startup validation: verify model max_tokens against provider limits",
    description="""At startup, validate that every model in MODEL_REGISTRY has a max_tokens
value that doesn't exceed the provider's actual limit. If litellm reports a lower
max_output_tokens than we configured, log a warning and cap to the provider limit.

Acceptance criteria:
- On startup, iterate MODEL_REGISTRY entries
- For each model, query litellm.get_model_info() for max_output_tokens
- If our configured max_tokens > provider limit, log warning and cap
- Add a validate_model_limits() function to llm_provider_types.py
- Unit tests covering: normal case, capped case, missing model info""",
)

ISSUE_438 = IssueSpec(
    issue_id="438",
    title="Consolidate model token limits: stages should use ModelConfig.max_tokens",
    description="""_get_model_max_output_tokens() in stages.py queries litellm at runtime and
maintains its own _max_output_cache. Since validate_model_limits() now caps
ModelConfig.max_tokens at startup, stages should just read from the registry.

The current code in src/accruvia/orchestration/ga_pipeline/stages.py:

    _max_output_cache: dict[str, int] = {}

    def _get_model_max_output_tokens(model_id: str) -> int:
        if model_id in _max_output_cache:
            return _max_output_cache[model_id]
        from accruvia.services.llm_provider_types import MODEL_REGISTRY, PROVIDER_MAX_TOKENS
        config = MODEL_REGISTRY.get(model_id)
        if not config:
            _max_output_cache[model_id] = 16384
            return 16384
        try:
            import litellm
            info = litellm.get_model_info(config.litellm_id)
            result = info.get("max_output_tokens", 16384)
        except Exception:
            result = 16384
        provider_cap = PROVIDER_MAX_TOKENS.get(config.provider)
        if provider_cap:
            result = min(result, provider_cap)
        _max_output_cache[model_id] = result
        return result

Replace this with a simple registry lookup:
- Read config.max_tokens from MODEL_REGISTRY (already capped at startup)
- Fall back to 16384 if model not in registry
- Remove _max_output_cache dict (registry IS the cache)
- Remove the litellm import from this function
- Remove the PROVIDER_MAX_TOKENS import from this function

Acceptance criteria:
- _get_model_max_output_tokens() returns ModelConfig.max_tokens from registry
- _max_output_cache module-level dict is removed
- No litellm import in _get_model_max_output_tokens
- No PROVIDER_MAX_TOKENS import in _get_model_max_output_tokens
- Default to 16384 for unknown models
- Unit tests: known model returns config.max_tokens, unknown model returns 16384""",
)

ISSUE_439 = IssueSpec(
    issue_id="439",
    title="API cost estimator for A/B test token data",
    description="""Add a cost estimation module that calculates what each A/B test run
would cost if the same token usage happened via API calls instead of CLI.

Create src/accruvia/telemetry/cost_model.py with:

1. PRICE_TABLE: dict mapping provider to per-token prices:
   - anthropic: input=$3.00/M, cached_input=$0.30/M, output=$15.00/M
   - openai: input=$2.50/M, cached_input=$1.25/M, output=$10.00/M
   - google: input=$1.25/M, cached_input=$0.315/M, output=$5.00/M
   - groq: input=$0.59/M, cached_input=$0.59/M, output=$0.79/M

2. estimate_cost(input_tokens, cached_tokens, output_tokens, provider) -> dict:
   Returns {"input_cost": float, "cached_cost": float, "output_cost": float,
   "total_cost": float, "savings_from_caching": float}
   where savings_from_caching = (cached_tokens * input_price - cached_tokens * cached_price)

3. format_cost_comparison(runs: list[dict]) -> str:
   Takes a list of run results (each with token breakdown and provider),
   returns a formatted string comparing costs across runs.
   Show per-run cost and highlight which mode is cheaper on API.

Acceptance criteria:
- PRICE_TABLE has entries for anthropic, openai, google, groq
- estimate_cost returns correct costs for each token type
- savings_from_caching shows how much caching saved vs full-price input
- format_cost_comparison produces readable output for 2+ runs
- Default provider is 'anthropic' when not specified
- Unit tests for all three functions""",
)

ISSUE_440 = IssueSpec(
    issue_id="440",
    title="Log full prompts, responses, and artifacts per A/B run",
    description="""The A/B runner saves result.json with token totals but not the actual
prompts, responses, or generated code. Without these, we can't debug why
a mode succeeded or failed, or compare solution quality across runs.

Create src/accruvia/telemetry/run_logger.py with:

1. RunLogger class:
   - __init__(self, out_dir: Path) — creates the output directory
   - log_iteration(self, iteration: int, prompt: str, raw_response: str,
     tokens: dict, validation: dict, artifacts: dict) — appends to iterations.jsonl
   - log_result(self, result: dict) — writes final result.json
   - log_diff(self, diff: str) — writes diff.patch
   - log_artifacts(self, artifacts: dict[str, str]) — writes each artifact
     to artifacts/{name}.py

2. Each iteration log entry in iterations.jsonl contains:
   - iteration number
   - full prompt sent
   - raw response received (truncated to 50KB if larger)
   - token usage
   - validation results per artifact
   - timestamp

Acceptance criteria:
- RunLogger creates out_dir on init
- log_iteration appends JSONL (one JSON object per line)
- log_artifacts writes files to artifacts/ subdirectory
- log_diff writes diff.patch
- Large responses are truncated to 50KB with a marker
- Unit tests for all methods""",
)

ISSUE_441 = IssueSpec(
    issue_id="441",
    title="Cost-aware model selection for pipeline stages",
    description="""Wire cost_model.py into live model routing so the pipeline picks
cheaper models when confidence is high, saving API spend on routine stages.

Changes required across multiple files:

1. src/accruvia/services/llm_provider_types.py:
   - Add cost_per_input_token and cost_per_output_token fields to ModelConfig
   - Populate from PRICE_TABLE values (per-million divided by 1M)
   - Models without pricing data default to the anthropic rate

2. src/accruvia/orchestration/ga_pipeline/orchestrator.py:
   - Add CostSelector class with select_model(stage, candidates, min_confidence=0.7):
     * Filters candidates to those with success_rate >= min_confidence
     * Among qualified candidates, picks the cheapest by cost_per_output_token
     * Falls back to GA selection if no candidate meets confidence threshold
   - Wire CostSelector into _choose_model() as an optional mode
   - Add enable_cost_selection flag to PipelineConfig (default False)

3. src/accruvia/telemetry/cost_model.py:
   - Add get_model_cost(model_id) -> dict returning per-token costs
   - Lookup from ModelConfig fields added in step 1

4. src/accruvia/orchestration/ga_pipeline/stages.py:
   - Each stage's execute() logs which selection method was used (GA vs cost)
   - Add selection_method field to stage result metadata

5. Tests:
   - CostSelector picks cheapest model above confidence threshold
   - CostSelector falls back to GA when no model meets threshold
   - get_model_cost returns correct rates for known models
   - Integration: PipelineConfig.enable_cost_selection toggles behavior

Acceptance criteria:
- CostSelector ranks models by cost when confidence is sufficient
- Falls back to GA exploration when confidence is low
- Pipeline stages log which selection method chose the model
- Feature is off by default (enable_cost_selection=False)
- Unit tests for CostSelector, get_model_cost, and config toggle""",
)

ISSUE_442 = IssueSpec(
    issue_id="442",
    title="Blame-driven feedback loop for GA settlement retraction",
    description="""When VERIFY fails, classify_failure() identifies which upstream stage
is to blame, but this blame is only logged — never acted upon. Close the
loop: retract the blamed stage's GA settlement and downgrade the model
that produced the faulty output.

Changes required across multiple files:

1. src/accruvia/orchestration/ga_pipeline/blame.py:
   - Add retraction_target(failure_classification) -> tuple[PipelineStage, str]
     that maps a blame classification to (stage, model_id) to retract
   - Enhance classify_failure() to return structured BlameResult with
     stage, model_id, confidence, and reason fields

2. src/accruvia/orchestration/ga_pipeline/settlement.py:
   - Add retract_settlement(stage, model_id, reason) method:
     * Finds the most recent win for that model on that stage
     * Reverses the fitness boost (subtract the original delta)
     * Logs the retraction with reason and timestamp
   - Add RetractionRecord dataclass for audit trail

3. src/accruvia/orchestration/ga_pipeline/ga_population.py:
   - Add downgrade_persona(model_id, penalty) method to ModelSelectionPopulation
   - penalty reduces the persona's fitness score (clamped to 0.0 minimum)
   - Persona is NOT removed — just deprioritized in next tournament selection

4. src/accruvia/orchestration/ga_pipeline/orchestrator.py:
   - In _handle_verify_failure(), after classify_failure():
     * Call retract_settlement() on the blamed stage
     * Call downgrade_persona() on the blamed model
     * Re-queue the blamed stage with fresh model selection
   - Add enable_blame_retraction flag to PipelineConfig (default False)

5. src/accruvia/orchestration/ga_pipeline/telemetry.py:
   - Log retraction events: stage, model, reason, fitness_before, fitness_after

6. Tests:
   - retract_settlement reverses a previous fitness boost
   - downgrade_persona reduces fitness but doesn't go below 0
   - _handle_verify_failure triggers retraction when blame confidence > 0.8
   - No retraction when blame confidence is low
   - RetractionRecord captures audit trail correctly

Acceptance criteria:
- BlameResult is a structured dataclass, not just a string
- retract_settlement undoes a specific prior settlement
- downgrade_persona penalizes but doesn't eliminate a model
- Orchestrator wires blame -> retraction -> re-queue
- Feature is off by default (enable_blame_retraction=False)
- Unit tests for all new methods""",
)

ISSUE_443 = IssueSpec(
    issue_id="443",
    title="Integrate trust registry into GA pipeline model selection",
    description="""The TrustGraduation system and GA populations are currently disconnected.
Wire trust scores into _choose_model() so stages prefer models with
proven track records when enough data exists.

Changes required across multiple files:

1. src/accruvia/orchestration/ga_pipeline/orchestrator.py:
   - Modify _choose_model() to query TrustGraduation before GA:
     * If a model has graduated (trust_level >= 'proven') for this stage,
       use it directly without GA tournament
     * If multiple models are graduated, pick the one with highest
       success_rate
     * Fall back to GA exploration if no graduated models exist
   - Add trust_weight parameter to PipelineConfig (0.0-1.0, default 0.0)
     controlling how much trust influences selection vs pure GA

2. src/accruvia/orchestration/ga_pipeline/trust_graduation.py:
   - Add get_graduated_models(stage: PipelineStage) -> list[TrustRecord]
   - Add query_trust(stage, model_id) -> TrustRecord | None
   - TrustRecord includes: model_id, trust_level, success_rate,
     total_attempts, last_success_at

3. src/accruvia/orchestration/ga_pipeline/stages.py:
   - Each stage's execute() records trust metadata in its result:
     selection_source ('trust' | 'ga' | 'cost'), model_id, trust_level
   - After successful execution, call trust_graduation.record_success()

4. src/accruvia/orchestration/ga_pipeline/models.py:
   - Add TrustRecord dataclass
   - Add selection_source field to stage result models

5. Tests:
   - _choose_model uses graduated model when trust_weight > 0
   - _choose_model falls back to GA when no graduated models
   - get_graduated_models returns only models above threshold
   - Stage results include selection_source metadata
   - trust_weight=0.0 disables trust-based selection entirely

Acceptance criteria:
- Trust-based selection is opt-in via trust_weight config
- Graduated models bypass GA tournament when trust_weight > 0
- Stage results record how the model was selected
- Falls back gracefully to GA when trust data is insufficient
- Unit tests for trust queries and selection logic""",
)

ISSUE_444 = IssueSpec(
    issue_id="444",
    title="Iterative code-fix loop instead of full regeneration on failure",
    description="""When CODE_WRITE fails validation (syntax errors, bad imports, undefined
names), the pipeline currently retries from scratch — regenerating ALL
code. This wastes tokens on large issues where only a small part is
wrong. Implement targeted fix-and-retry.

Changes required across multiple files:

1. src/accruvia/orchestration/ga_pipeline/stages.py (CodeWriteStage):
   - After validation failure, instead of full retry:
     a. Save the failed code to a staging buffer
     b. Collect specific errors from ImportValidator and NameChecker
     c. Build a fix prompt: "Here is the code you generated and the
        errors found. Fix ONLY the errors, preserve everything else."
     d. Send fix prompt to LLM (may use a different/cheaper model)
     e. Validate the fixed code
     f. Allow up to 3 fix attempts before falling back to full regen
   - Add FixAttempt dataclass: attempt_num, errors_in, code_out,
     errors_remaining, model_used, tokens_used
   - Track fix attempts separately from full regeneration attempts

2. src/accruvia/orchestration/import_validator.py:
   - Add validate_with_details(code, filename) -> ValidationResult
   - ValidationResult includes: valid (bool), errors (list of
     ImportError with line_number, module_name, error_type)
   - Current validate() becomes a thin wrapper over validate_with_details()

3. src/accruvia/orchestration/name_checker.py:
   - Add check_with_details(code) -> NameCheckResult
   - NameCheckResult includes: valid (bool), errors (list of
     UndefinedName with line_number, name, context)
   - Current check() becomes a thin wrapper over check_with_details()

4. src/accruvia/orchestration/ga_pipeline/models.py:
   - Add FixAttempt dataclass
   - Add ValidationResult and NameCheckResult dataclasses
   - Add max_fix_attempts field to PipelineConfig (default 3)

5. src/accruvia/orchestration/ga_pipeline/telemetry.py:
   - Log fix attempts: iteration, attempt_num, errors_before,
     errors_after, tokens_used, model_used
   - Distinguish fix-tokens from regen-tokens in reporting

6. Tests:
   - Fix prompt includes the specific errors and original code
   - Fix attempt succeeds when LLM corrects the error
   - Falls back to full regen after max_fix_attempts exhausted
   - validate_with_details returns structured error info
   - check_with_details returns structured error info
   - Fix tokens tracked separately from regen tokens
   - FixAttempt dataclass captures all metadata

Acceptance criteria:
- CODE_WRITE tries targeted fixes before full regeneration
- Fix prompts include specific error messages and line numbers
- Up to max_fix_attempts fixes before falling back
- ImportValidator and NameChecker provide structured error details
- Fix vs regen token usage tracked separately
- Unit tests for fix loop, structured validation, and telemetry""",
)

ISSUE_445 = IssueSpec(
    issue_id="445",
    title="Task fingerprinting for issue complexity stratification",
    description="""Extract complexity features from issues before screening so A/B results
can be stratified by issue size. Small vs large issues have different
cost profiles — we need the data to prove it.

Changes required across multiple files:

1. src/accruvia/orchestration/ga_pipeline/stages.py:
   - Add FeatureExtractionStage (or hook into ScreeningStage.execute()):
     * Parse issue description for: file count hints, import references,
       class/function mentions, estimated lines of change
     * Assign complexity_bucket: 'small' (1-2 files, <100 LOC),
       'medium' (3-5 files, 100-300 LOC), 'large' (6+ files, 300+ LOC)
   - Output TaskFingerprint attached to PipelineIssue

2. src/accruvia/orchestration/ga_pipeline/models.py:
   - Add TaskFingerprint dataclass: file_count_estimate (int),
     loc_estimate (int), import_count (int), complexity_bucket (str),
     cross_cutting (bool), extracted_at (datetime)
   - Add fingerprint field to PipelineIssue (Optional[TaskFingerprint])

3. src/accruvia/orchestration/ga_pipeline/trust_graduation.py:
   - Refine TaskFingerprint calculation using extracted features
   - Use fingerprint in trust lookups: "this model is trusted for
     small issues but not large ones"

4. src/accruvia/orchestration/ga_pipeline/telemetry.py:
   - Record fingerprint in pipeline_runs table
   - Add fingerprint columns: file_count, loc_estimate, complexity_bucket

5. scripts/ab_report.py:
   - Group results by complexity_bucket when fingerprint data exists
   - Show cost comparison per bucket: "small issues: schema saves X%,
     large issues: schema saves Y%"

6. Tests:
   - Feature extraction parses file count from issue description
   - complexity_bucket assigned correctly for each size range
   - TaskFingerprint attached to PipelineIssue after extraction
   - ab_report groups by bucket when data available
   - Missing fingerprint data handled gracefully (no crash)

Acceptance criteria:
- TaskFingerprint captures complexity signals from issue text
- complexity_bucket categorizes issues into small/medium/large
- Fingerprint stored in telemetry for post-hoc analysis
- ab_report can stratify by complexity when data exists
- Unit tests for extraction, bucketing, and report grouping""",
)

ISSUE_446 = IssueSpec(
    issue_id="446",
    title="AB runner: add max iteration cap and failure return path",
    description="""Both run_schema_mode() and run_direct_mode() have `while True` loops
with no exit condition. On hard issues the runner loops forever, burning
tokens and producing garbage cost data. This blocks all larger issues.

Fix in scripts/ab_runner.py:

1. Add MAX_ITERATIONS constant (default 10) at module level.

2. In run_schema_mode() (line ~779 `while True`):
   - Change to `while iteration < MAX_ITERATIONS`
   - After the loop, return a failure result dict:
     {"mode": "schema", "issue_id": ..., "run_id": ..., "success": False,
      "total_tokens": aggregate_tokens(iterations),
      "total_time_s": time.time() - t0,
      "iterations": iteration, "per_iteration": iterations}
   - Print "[MAX ITERATIONS] Giving up after {iteration} iterations"

3. In run_direct_mode() (line ~905 `while True`):
   - Same change: `while iteration < MAX_ITERATIONS`
   - Same failure return dict with mode="direct"
   - Same max iteration message

4. Tests:
   - Schema mode returns success=False after MAX_ITERATIONS
   - Direct mode returns success=False after MAX_ITERATIONS
   - Failure result includes correct token aggregation
   - save_ab_result handles success=False results

Acceptance criteria:
- Both loops exit after MAX_ITERATIONS
- Failure results are saved with success=False
- Token totals are correct even on failure
- ab_report.py handles failed runs in its stats""",
)

ISSUE_447 = IssueSpec(
    issue_id="447",
    title="AB runner: fix code_schema double-parse crash risk",
    description="""In run_schema_mode(), code_schema_dict is populated in two places with
inconsistent type handling:

Line ~818: `cs = code_schema_dict or json.loads(artifacts.get("code_schema", "{}"))`
  - artifacts["code_schema"] may already be a dict from parse_schema_response()
  - json.loads(dict) would raise TypeError

Line ~844: `code_schema_dict = json.loads(content) if isinstance(content, str) else content`
  - This correctly handles both types but runs AFTER the line 818 path

Fix in scripts/ab_runner.py:

1. In the import allowlist block (~line 816-825):
   - Use the same isinstance guard: parse only if content is a string
   - Or restructure so code_schema_dict is set before the allowlist block

2. Tests:
   - Schema mode handles code_schema as dict without crash
   - Schema mode handles code_schema as JSON string without crash
   - Import allowlist is correctly derived in both cases

Acceptance criteria:
- No TypeError when code_schema arrives as a dict
- Import allowlist works regardless of artifact type
- Unit test covers both code paths""",
)

ISSUE_448 = IssueSpec(
    issue_id="448",
    title="Move AB runner and telemetry into accruvia_client package",
    description="""The AB runner (scripts/ab_runner.py), telemetry modules
(src/accruvia/telemetry/cost_model.py, run_logger.py), and report script
(scripts/ab_report.py) ARE the client product — they resolve issues, validate
code, track costs, and report results. But they live in scripts/ and a generic
telemetry package instead of accruvia_client/ where they belong.

This is a pure move — no logic changes, no refactoring. Fix imports, verify
tests pass.

Steps:

1. Move scripts/ab_runner.py → src/accruvia_client/runner.py
   - Update all imports (sys.path hacks → proper package imports)
   - Add CLI entry point in src/accruvia_client/__main__.py or keep a
     thin scripts/ab_runner.py that imports and calls main()

2. Move scripts/ab_report.py → src/accruvia_client/report.py
   - Same import cleanup

3. Move src/accruvia/telemetry/cost_model.py → src/accruvia_client/telemetry/cost_model.py
   - Update imports in runner.py and report.py

4. Move src/accruvia/telemetry/run_logger.py → src/accruvia_client/telemetry/run_logger.py
   - Update imports in runner.py

5. Update test imports:
   - tests/test_ab_runner.py → import from accruvia_client.runner
   - tests/test_cost_model.py → import from accruvia_client.telemetry.cost_model
   - tests/test_run_logger.py → import from accruvia_client.telemetry.run_logger

6. Keep thin wrapper scripts in scripts/ for CLI convenience:
   - scripts/ab_runner.py: 3 lines — import and call main()
   - scripts/ab_report.py: 3 lines — import and call main()

Acceptance criteria:
- All existing tests pass with updated imports
- ab_runner CLI still works: PYTHONPATH=src python scripts/ab_runner.py --issue 446 --mode schema
- No logic changes — pure file relocation + import fixup
- accruvia_client/ is the home for all client-side functionality""",
)

ISSUE_449 = IssueSpec(
    issue_id="449",
    title="Refactor runner monolith into client modules",
    description="""After issue #448 moves ab_runner.py into accruvia_client/runner.py,
it's still a ~1500-line monolith. Split it into focused modules that reflect
the product's actual architecture.

Depends on: #448 (move first, refactor second)

Target module structure under src/accruvia_client/:

1. resolver.py — The core issue resolution engine:
   - run_schema_mode() and run_direct_mode()
   - IssueSpec dataclass
   - MAX_ITERATIONS constant
   - build_schema_prompt(), build_direct_prompt()
   - parse_schema_response(), discover_artifacts()

2. validator.py — Artifact validation pipeline:
   - validate_artifact() (ast.parse + ImportValidator + NameChecker)
   - _check_has_logic(), _coerce_code_schema()
   - ArtifactBank (accept/reject/eject state machine)
   - blame_pytest()

3. reporter.py — Reporting and cost analysis:
   - Moved from report.py
   - format_cost_comparison integration
   - Per-run and per-issue aggregation

4. telemetry/ — Already moved in #448:
   - cost_model.py (API cost estimation)
   - run_logger.py (iteration/artifact logging)

5. issues.py — Issue registry:
   - ISSUES dict
   - All IssueSpec definitions (437-449+)
   - Separated from runner logic so issues can be added without
     touching the resolver

6. runner.py — Thin orchestration layer:
   - CLI argument parsing
   - Worktree setup/teardown
   - Calls resolver with config
   - Saves results via run_logger

Each module should be independently testable. Existing tests split
accordingly into test_resolver.py, test_validator.py, etc.

Acceptance criteria:
- No module exceeds 400 lines
- Each module has a single responsibility
- All existing tests pass (relocated to match new modules)
- CLI entry point unchanged
- No logic changes — pure structural refactor""",
)

ISSUE_456 = IssueSpec(
    issue_id="456",
    title="QA review panel: single-call multi-persona code review before VERIFY",
    description="""Add an automated QA gate that reviews accepted artifacts before VERIFY
runs. A single LLM call evaluates the code from multiple stakeholder
perspectives, catching issues that deterministic validation misses.

Depends on: #448 (move to accruvia_client first)

The QA panel runs identically for both schema and direct modes — it sees
only the accepted artifacts, not how they were generated. This preserves
AB test fairness while adding quality signal.

Architecture:

1. src/accruvia_client/qa_panel.py — QAPanel class:

   - __init__(self, model: str = "claude-haiku-4.5") — cheap model for reviews
   - review(self, issue: IssueSpec, artifacts: dict[str, str]) -> QAVerdict
   - Single LLM call with a system prompt containing ALL reviewer personas:

     System prompt structure:
     ```
     You are a QA review panel. Evaluate this code from these perspectives:

     ## Security Reviewer
     Check for: injection risks, hardcoded secrets, unsafe deserialization,
     unvalidated input at system boundaries, OWASP top 10.

     ## Efficiency Reviewer
     Check for: O(n²) in hot paths, unnecessary allocations, redundant
     iterations, missing early returns, unbounded growth.

     ## Factorability Reviewer
     Check for: functions >50 lines, classes with >1 responsibility,
     copy-paste patterns, missing abstractions that would reduce duplication.

     ## Product Owner
     Check for: does the code actually solve the issue as described?
     Are acceptance criteria met? Any scope creep or missing requirements?

     ## UX Reviewer (API surface)
     Check for: confusing parameter names, inconsistent return types,
     missing defaults, surprising behavior, poor error messages.
     ```

   - Response schema (JSON):
     ```json
     {
       "approved": bool,
       "hard_rejects": [{"reviewer": str, "concern": str, "line": int|null}],
       "suggestions": [{"reviewer": str, "suggestion": str}],
       "summary": str
     }
     ```

   - QAVerdict dataclass:
     approved (bool), hard_rejects (list), suggestions (list), summary (str),
     tokens_used (int), model_used (str)

2. Integration into runner (both modes equally):

   Insert point: after bank.all_accepted() but before write_artifacts_to_disk()

   ```python
   if qa_panel is not None:
       verdict = qa_panel.review(issue, bank.get_all_accepted())
       if not verdict.approved:
           for reject in verdict.hard_rejects:
               blamed = "source_code"
               bank.eject(blamed, f"QA {reject['reviewer']}: {reject['concern']}")
           continue
       log(f"  [QA] Approved with {len(verdict.suggestions)} suggestions")
   ```

   Both modes hit this identical code path — QA doesn't know or care
   whether artifacts came from schema parsing or file discovery.

3. Configuration:
   - --qa flag on CLI to enable (off by default, doesn't affect existing runs)
   - qa_model parameter for model selection (default: cheapest available)
   - max_qa_rejections: int = 2 — don't let QA loop forever

4. Telemetry:
   - QA tokens tracked separately from generation tokens
   - QA verdicts logged in iterations.jsonl via RunLogger
   - ab_report shows QA pass/fail rate per mode

5. Tests:
   - QAPanel.review returns QAVerdict with correct structure
   - Approved verdict doesn't block VERIFY
   - Hard reject ejects artifact and triggers retry
   - max_qa_rejections prevents infinite QA loop
   - QA tokens are not counted in generation token totals
   - Both modes produce identical QA inputs for same artifacts
   - QA panel uses cheap model (not the generation model)

Acceptance criteria:
- Single LLM call per QA review (not N separate calls)
- All 5 reviewer perspectives in one system prompt
- Identical QA path for both schema and direct modes
- QA is opt-in (--qa flag), default off
- QA tokens tracked separately from generation tokens
- Hard rejects feed back through existing retry mechanism
- Suggestions logged but don't block acceptance
- Unit tests for QAPanel, verdict handling, and telemetry""",
)

ISSUES = {
    "437": ISSUE_437,
    "438": ISSUE_438,
    "439": ISSUE_439,
    "440": ISSUE_440,
    "441": ISSUE_441,
    "442": ISSUE_442,
    "443": ISSUE_443,
    "444": ISSUE_444,
    "445": ISSUE_445,
    "446": ISSUE_446,
    "447": ISSUE_447,
    "448": ISSUE_448,
    "449": ISSUE_449,
    "450": ISSUE_456,
    "456": ISSUE_456,
}


def load_issue_from_github(issue_id: str, base_dir: Path) -> IssueSpec | None:
    """Fetch issue details from GitHub when the local registry is stale."""
    try:
        result = subprocess.run(
            ["gh", "issue", "view", issue_id, "--json", "number,title,body,labels"],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    title = payload.get("title")
    description = payload.get("body") or ""
    if not title:
        return None

    labels = []
    raw_labels = payload.get("labels") or []
    for label in raw_labels:
        if isinstance(label, dict):
            name = label.get("name")
            if name:
                labels.append(name)
        elif isinstance(label, str):
            labels.append(label)

    return IssueSpec(
        issue_id=str(payload.get("number") or issue_id),
        title=title,
        description=description,
        labels=labels or None,
    )
