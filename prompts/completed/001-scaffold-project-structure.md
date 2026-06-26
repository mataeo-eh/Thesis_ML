<objective>
Scaffold the repository structure for an SC2 strategy-prediction system built on masked discrete diffusion. This is the foundational prompt: it establishes directory layout, coding conventions (CLAUDE.md), the single YAML config and its validating dataclass, and a placeholder special-token registry. No model, data, or training logic is built here — this prompt creates the skeleton every later prompt builds on.

The repository consumes output from a separate, already-complete extraction repo (`SC2-gamestate-extractor`). This repo is the modeling side.
</objective>

<context>
- Python + PyTorch project. Tests via pytest.
- The architectural source of truth is `./SPEC.md` at the repo root. READ IT IN FULL before doing anything. It defines settled decisions, provisional config parameters, open questions you must NOT resolve, and a hard ban list. On any conflict between SPEC.md and any other instruction, SPEC.md wins.
- This is a localized scaffolding task — single agent, no sub-agent orchestration.
- Do not install heavy dependencies or build environment tooling beyond what scaffolding needs. Keep it minimal: a direct `pip install` of the few packages required, recorded in a requirements file. No Docker, no Conda environment files, no CI config in this prompt.
</context>

<requirements>
1. Read `./SPEC.md` fully. Treat §11 as the authoritative list of config parameters and their provisional defaults. Treat §14 as forbidden. Treat §15 as the directory convention.

2. Create the directory structure from SPEC.md §15:
   - `./prompts/` and `./prompts/completed/`
   - `./research/`, `./plans/`, `./diagnostics/`
   - `./tests/` and `./tests/fixtures/`
   - `./src/<pkg>/` — use a `src/` layout (the package lives under `src/`, not at the repo root). Document the package name in CLAUDE.md.
   - Add a `.gitkeep` to any directory that would otherwise be empty so the structure survives version control.

3. Create `CLAUDE.md` at the repo root carrying CODING conventions only (architecture truth lives in SPEC.md and must not be duplicated). It must state:
   - Python version target and PyTorch as the framework.
   - Test framework is pytest; tests live in `./tests/`.
   - Configuration is a single YAML file validated by a dataclass; parameters are never hardcoded — they are read from config.
   - The directory layout and what each directory is for.
   - A pointer to SPEC.md as the architecture source of truth, with an explicit note that SPEC.md wins on any conflict.
   - A short "do not" section reminding agents not to implement anything from SPEC.md §14, and not to resolve anything from SPEC.md §12.

4. Create the single configuration system:
   - One YAML file (e.g. `./config/default.yaml`) containing every PROVISIONAL parameter from SPEC.md §11 with its default value, organized into readable sections (data, model, fog, sampler, loss).
   - One dataclass-based config loader (e.g. `./<pkg>/config.py`) that parses the YAML into a typed, validated dataclass. Validation must reject unknown keys and wrong types, and must fail loudly with a clear message naming the offending field. Nested config sections may be nested dataclasses.
   - The `fog_rate_distribution` and `mask_schedule` fields will hold structured values (e.g. a distribution name plus parameters); represent them in a way the dataclass can validate, not as free-form strings to be parsed later.

5. Create a special-token registry (e.g. `./<pkg>/vocab/special_tokens.py`) that defines, as named constants, every special token from SPEC.md §4: `[MASK]`, `[PAD]`, `[END]`, `[DELIMITER]`, `[WIN]`, `[LOSS]`. Assign them stable integer IDs reserved at the low end of the vocabulary. This is ONLY the special-token reservation — content tokens are derived from the extractor schema in a later prompt and must NOT be invented here. Leave a documented gap/offset where content-token IDs will begin.

6. Create a `pyproject.toml` configured for `uv` as the project/dependency manager and a `src/` layout. List only what scaffolding and the config loader need right now (PyYAML, the dataclass-validation approach you chose, pytest, torch). Do not pre-add packages later prompts will need; they can add their own via `uv add`. Verify the current `uv` + `pyproject.toml` conventions before writing the file — confirm the current `[project]` and build-backend syntax and the correct way to declare a `src/` package, since this tooling changes.
</requirements>

<implementation>
- Before writing the config loader, verify the current recommended approach for the validation library you choose. If you use Pydantic, confirm the current major version's API (v2 syntax differs from v1). If you use plain dataclasses plus manual validation, that is acceptable and has no version risk — prefer it if it keeps things simple. State which you chose and why in a one-line comment.
- The config dataclass is the contract every later prompt depends on. Field names should match SPEC.md §11 parameter names where reasonable so cross-referencing is trivial.
- Keep the special-token IDs and the documented content-token offset in one obvious place — later tokenization work will read this.
- Do not write model code, dataset code, training code, or any logic beyond scaffolding + config + token registry. Those are later prompts.
- Avoid premature abstraction: no base classes, plugin systems, or registries beyond the special-token constants. Build the skeleton, nothing more.
</implementation>

<output>
Create, using relative paths:
- `./CLAUDE.md` — coding conventions (not architecture)
- `./config/default.yaml` — all §11 parameters with defaults
- `./src/<pkg>/config.py` — dataclass config loader with validation
- `./src/<pkg>/vocab/special_tokens.py` — special-token constants + reserved IDs + documented content-token offset
- `./pyproject.toml` — uv-managed, src/ layout, minimal dependencies
- Directory structure per SPEC.md §15, with `.gitkeep` files where needed
- `./tests/test_config.py` — tests proving config validation works (see verification)
</output>

<verification>
Before declaring complete, you MUST run these checks and report each as PASS/FAIL with the command run and its result:

1. **Config loads:** Run a script (or `python -c`) that imports the config loader and loads `./config/default.yaml` successfully. Confirm every §11 parameter is present on the resulting dataclass with the SPEC default value. PASS only if all parameters load with correct defaults.

2. **Validation rejects bad input:** Run `pytest ./tests/test_config.py`. The tests must include: (a) a valid config loads; (b) an unknown key is rejected with an error naming the key; (c) a wrong-typed value is rejected with an error naming the field. PASS only if all three tests pass.

3. **Special tokens importable and distinct:** Run a script that imports the special-token registry and asserts all six special tokens have distinct integer IDs and that the documented content-token offset is strictly greater than every special-token ID. PASS only if assertions hold.

4. **Directory structure exists:** List the tree and confirm every directory from SPEC.md §15 exists. PASS only if all are present.

For each check: state what you ran, the result, and PASS/FAIL. If any check fails, fix it and re-run ALL checks before declaring complete. Do not skip verification. Do not declare success without running every check.
</verification>

<success_criteria>
- `./SPEC.md` was read before any work began; nothing from §14 was implemented and nothing from §12 was resolved.
- Repository directory structure matches SPEC.md §15.
- `CLAUDE.md` exists, carries coding conventions only, and points to SPEC.md as architecture truth.
- A single YAML config holds every §11 parameter at its provisional default; a dataclass loader parses and validates it, rejecting unknown keys and wrong types with field-naming errors.
- The six special tokens from §4 exist as named constants with distinct reserved IDs and a documented offset for future content tokens.
- A minimal `pyproject.toml` (uv-managed, src/ layout) lists only what is needed now.
- All four verification checks PASS.
</success_criteria>
