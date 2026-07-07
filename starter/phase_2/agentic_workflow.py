# agentic_workflow.py
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "workflow_config.json"
PRODUCT_SPEC_PATH = BASE_DIR / "Product-Spec-Email-Router.txt"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_workflow_config(config_path=CONFIG_PATH):
    """Load configurable personas, route descriptions, and evaluation criteria."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def load_product_spec(product_spec_path=PRODUCT_SPEC_PATH):
    """Load the Email Router product specification used by the pilot workflow."""
    with open(product_spec_path, "r", encoding="utf-8") as product_spec_file:
        return product_spec_file.read()


def require_api_key():
    """Load the OpenAI API key from the local environment."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file or environment.")
    return openai_api_key


def evaluate_generated_artifact(evaluation_agent, generated_artifact):
    """
    Evaluate an artifact that has already been generated.

    The original project pattern calls evaluate() with a prompt, which makes the worker
    generate again. This helper keeps the rubric-compatible EvaluationAgent while avoiding
    double generation in the orchestrated workflow.
    """
    if hasattr(evaluation_agent, "evaluate_artifact"):
        return evaluation_agent.evaluate_artifact(generated_artifact)
    return evaluation_agent.evaluate(generated_artifact)


def validate_required_markers(name, content, required_markers):
    """Run deterministic format checks for required section markers."""
    missing_markers = [marker for marker in required_markers if marker not in content]
    return {
        "name": name,
        "passed": not missing_markers,
        "missing_markers": missing_markers,
    }


def build_workflow(openai_api_key, product_spec, config):
    """Instantiate the planning, role, evaluation, and routing agents."""
    roles = config["roles"]
    completed_steps = []

    action_planning_agent = ActionPlanningAgent(
        openai_api_key,
        config["knowledge_action_planning"],
    )

    product_manager_config = roles["product_manager"]
    program_manager_config = roles["program_manager"]
    development_engineer_config = roles["development_engineer"]

    product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key,
        product_manager_config["persona"],
        f"{product_manager_config['knowledge']}\n\nProduct specification:\n{product_spec}",
    )
    product_manager_evaluation_agent = EvaluationAgent(
        openai_api_key=openai_api_key,
        persona=product_manager_config["evaluation_persona"],
        evaluation_criteria=product_manager_config["evaluation_criteria"],
        worker_agent=product_manager_knowledge_agent,
        max_interactions=10,
    )

    program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key,
        program_manager_config["persona"],
        program_manager_config["knowledge"],
    )
    program_manager_evaluation_agent = EvaluationAgent(
        openai_api_key=openai_api_key,
        persona=program_manager_config["evaluation_persona"],
        evaluation_criteria=program_manager_config["evaluation_criteria"],
        worker_agent=program_manager_knowledge_agent,
        max_interactions=10,
    )

    development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key,
        development_engineer_config["persona"],
        development_engineer_config["knowledge"],
    )
    development_engineer_evaluation_agent = EvaluationAgent(
        openai_api_key=openai_api_key,
        persona=development_engineer_config["evaluation_persona"],
        evaluation_criteria=development_engineer_config["evaluation_criteria"],
        worker_agent=development_engineer_knowledge_agent,
        max_interactions=10,
    )

    def product_manager_support_function(query):
        response_from_knowledge_agent = product_manager_knowledge_agent.respond(query)
        evaluation_result = evaluate_generated_artifact(
            product_manager_evaluation_agent,
            response_from_knowledge_agent,
        )
        logger.info("Product Manager evaluation result: %s", evaluation_result.get("scorecard"))
        return evaluation_result["final_response"]

    def program_manager_support_function(query):
        user_stories = completed_steps[0] if len(completed_steps) > 0 else ""
        full_query = f"{query}\n\nHere are the user stories to organize into features:\n{user_stories}"
        response_from_knowledge_agent = program_manager_knowledge_agent.respond(full_query)
        evaluation_result = evaluate_generated_artifact(
            program_manager_evaluation_agent,
            response_from_knowledge_agent,
        )
        logger.info("Program Manager evaluation result: %s", evaluation_result.get("scorecard"))
        return evaluation_result["final_response"]

    def development_engineer_support_function(query):
        user_stories = completed_steps[0] if len(completed_steps) > 0 else ""
        features = completed_steps[1] if len(completed_steps) > 1 else ""
        full_query = (
            f"{query}\n\n"
            f"Here are the user stories:\n{user_stories}\n\n"
            f"Here are the features:\n{features}"
        )
        response_from_knowledge_agent = development_engineer_knowledge_agent.respond(full_query)
        evaluation_result = evaluate_generated_artifact(
            development_engineer_evaluation_agent,
            response_from_knowledge_agent,
        )
        logger.info("Development Engineer evaluation result: %s", evaluation_result.get("scorecard"))
        return evaluation_result["final_response"]

    routes = [
        {
            "name": product_manager_config["display_name"],
            "description": product_manager_config["route_description"],
            "func": product_manager_support_function,
        },
        {
            "name": program_manager_config["display_name"],
            "description": program_manager_config["route_description"],
            "func": program_manager_support_function,
        },
        {
            "name": development_engineer_config["display_name"],
            "description": development_engineer_config["route_description"],
            "func": development_engineer_support_function,
        },
    ]

    routing_agent = RoutingAgent(openai_api_key, routes)

    return action_planning_agent, routing_agent, completed_steps


def validate_completed_steps(completed_steps, config):
    """Validate generated role outputs using deterministic marker checks."""
    role_order = ["product_manager", "program_manager", "development_engineer"]
    validation_results = []

    for index, role_name in enumerate(role_order):
        role_config = config["roles"][role_name]
        content = completed_steps[index] if index < len(completed_steps) else ""
        validation_results.append(
            validate_required_markers(
                role_config["display_name"],
                content,
                role_config.get("required_markers", []),
            )
        )

    return validation_results


def format_validation_summary(validation_results):
    """Format deterministic validation results for terminal output."""
    lines = ["\n## Deterministic Format Checks"]
    for result in validation_results:
        if result["passed"]:
            lines.append(f"- {result['name']}: passed")
        else:
            missing = ", ".join(result["missing_markers"])
            lines.append(f"- {result['name']}: missing {missing}")
    return "\n".join(lines)


def format_final_project_plan(completed_steps, validation_results=None):
    """Format the workflow outputs into the final Email Router project plan."""
    if len(completed_steps) >= 3:
        final_plan = f"""
# Email Router Development Plan

## 1. User Stories
{completed_steps[0]}

## 2. Product Features
{completed_steps[1]}

## 3. Engineering Tasks
{completed_steps[2]}
""".strip()
        if validation_results:
            final_plan += "\n" + format_validation_summary(validation_results)
        return final_plan

    if completed_steps:
        partial_output = ["Partial workflow output:"]
        for idx, item in enumerate(completed_steps, start=1):
            partial_output.append(f"\n--- Completed Step {idx} ---\n{item}")
        if validation_results:
            partial_output.append(format_validation_summary(validation_results))
        return "\n".join(partial_output)

    return "No steps were executed."


def run_workflow(action_planning_agent, routing_agent, completed_steps, workflow_prompt):
    """Run the planner-router workflow and collect each role output."""
    print("\nDefining workflow steps from the workflow prompt")
    workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

    for idx, step in enumerate(workflow_steps):
        print(f"\n--- Executing Step {idx+1}/{len(workflow_steps)}: {step} ---")
        result = routing_agent.route(step)
        completed_steps.append(result)
        print(f"Result for Step '{step}':\n{result}")

    return completed_steps


def main():
    """Run the Email Router project-planning workflow."""
    openai_api_key = require_api_key()
    config = load_workflow_config()
    product_spec = load_product_spec()

    action_planning_agent, routing_agent, completed_steps = build_workflow(
        openai_api_key,
        product_spec,
        config,
    )

    print("\n*** Workflow execution started ***\n")
    workflow_prompt = config["workflow_prompt"]
    print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

    completed_steps = run_workflow(
        action_planning_agent,
        routing_agent,
        completed_steps,
        workflow_prompt,
    )
    validation_results = validate_completed_steps(completed_steps, config)

    print("\n*** Final Email Router Project Plan ***")
    print(format_final_project_plan(completed_steps, validation_results))


if __name__ == "__main__":
    main()
