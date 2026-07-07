# Agentic Project Planner

Agentic Project Planner is a Python implementation of a reusable agent workflow for turning a product specification into a structured development plan. The pilot input is an Email Router product spec, and the workflow produces user stories, product features, and engineering tasks.

This started from an AI workflows project, but the repo is organized as a portfolio case study in agent orchestration, semantic routing, evaluation loops, and workflow reliability.

## What this demonstrates

- A reusable agent library with direct prompting, persona prompting, knowledge-augmented prompting, RAG, evaluation, routing, and action-planning agents.
- A multi-role planning workflow that simulates Product Manager, Program Manager, and Development Engineer responsibilities.
- Semantic routing with OpenAI embeddings and cached route-description embeddings.
- LLM-based evaluation plus deterministic marker checks for required output sections.
- Config-driven workflow roles, route descriptions, personas, and evaluation criteria.
- Importable workflow orchestration through `main()` and `if __name__ == "__main__"`.

## Repository structure

```text
.
├── requirements.txt
├── starter/
│   ├── phase_1/
│   │   ├── workflow_agents/
│   │   │   └── base_agents.py
│   │   ├── direct_prompt_agent.py
│   │   ├── augmented_prompt_agent.py
│   │   ├── knowledge_augmented_prompt_agent.py
│   │   ├── rag_knowledge_prompt_agent.py
│   │   ├── evaluation_agent.py
│   │   ├── routing_agent.py
│   │   └── action_planning_agent.py
│   └── phase_2/
│       ├── Product-Spec-Email-Router.txt
│       ├── agentic_workflow.py
│       ├── workflow_config.json
│       └── workflow_agents/
│           └── base_agents.py
```

## Agent library

The core agent classes live in `workflow_agents/base_agents.py`:

- `DirectPromptAgent` sends the user prompt directly to the model.
- `AugmentedPromptAgent` adds a persona-defining system prompt.
- `KnowledgeAugmentedPromptAgent` constrains the response to explicit supplied knowledge.
- `RAGKnowledgePromptAgent` chunks supplied text, embeds it, retrieves the closest chunk, and answers from retrieved context.
- `EvaluationAgent` evaluates worker output against criteria and can now score already-generated artifacts without forcing a second generation pass.
- `RoutingAgent` routes work to the best role agent using embedding similarity, with logging and embedding caching.
- `ActionPlanningAgent` turns a high-level workflow prompt into ordered action steps.

## Phase 2 workflow

`starter/phase_2/agentic_workflow.py` runs the Email Router project-planning workflow:

1. Load the Email Router product spec.
2. Load personas, criteria, and route descriptions from `workflow_config.json`.
3. Ask the `ActionPlanningAgent` for workflow steps.
4. Route each step to the appropriate role agent.
5. Evaluate each generated artifact.
6. Print a consolidated final project plan containing:
   - user stories;
   - product features;
   - engineering tasks; and
   - deterministic format-check results.

The final report is printed as a complete plan instead of only printing the final routed step.

## Running locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Set the API key in your shell or `.env` file:

```bash
OPENAI_API_KEY=your_key_here
```

Run the main workflow:

```bash
cd starter/phase_2
python agentic_workflow.py
```

Run individual Phase 1 agent scripts from `starter/phase_1` when validating the reusable agent classes.

## Configuration

The Phase 2 workflow reads from `starter/phase_2/workflow_config.json`. You can modify role personas, evaluation criteria, route descriptions, deterministic required markers, and the workflow prompt without changing the Python orchestration code.

## Reliability notes

This repo keeps the original assignment-compatible agent patterns but adds several portfolio-oriented improvements:

- route-description embedding caching;
- logging around route candidates, selected agents, scorecards, and output length;
- OpenAI timeout and retry configuration through shared client construction;
- deterministic marker checks for final user-story, feature, and task formats; and
- a `main()` entry point so the workflow can be imported, tested, or reused.

The workflow still returns role outputs as plain text because that matches the project rubric. A natural next improvement would be to ask each role agent for structured JSON and validate it with Pydantic before formatting the final report.

## Project context

This project was built from an educational AI workflows assignment. The repo is maintained as a learning and portfolio artifact focused on agent design, workflow orchestration, and evaluation-driven planning.
