# Project Reflection

## What works well

The project separates reusable agent patterns from the final Email Router workflow. Phase 1 demonstrates individual agent behaviors, while Phase 2 composes action planning, semantic routing, role-specific knowledge agents, and evaluation agents into a full project-planning pipeline.

The workflow is now easier to maintain because personas, route descriptions, evaluation criteria, and deterministic required markers live in `starter/phase_2/workflow_config.json`. The script can also be imported safely because execution is wrapped in a `main()` function.

## Reliability improvements added

- The final workflow output now prints a consolidated Email Router development plan instead of only the last completed step.
- The router logs each candidate route and selected agent.
- Route-description embeddings are cached so repeated calls do not recompute static route embeddings.
- OpenAI clients share timeout and retry configuration.
- Evaluation results include a simple deterministic scorecard in addition to the LLM-based judgment.
- Final output includes deterministic marker checks for user stories, product features, and engineering tasks.

## Limitations

The role agents still return plain text because that matches the project rubric. Plain text is easy to read, but it is harder to validate and route downstream than structured JSON.

The evaluation loop is still mostly LLM-driven. Deterministic marker checks help catch missing sections, but they do not fully verify semantic quality, feasibility, or traceability to the product spec.

## Best next step

The strongest next improvement would be a final collaborative review stage. After the Product Manager, Program Manager, and Development Engineer agents produce the plan, a final review agent could ask all three roles to critique the complete output before publishing the final plan.

A deeper version of that improvement would have each role return structured JSON, validate it with Pydantic models, and then generate the final Markdown report from validated data rather than free-form text.
