# base_agents.py
import csv
import logging
import os
import re
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from openai import OpenAI

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

logger = logging.getLogger(__name__)


def make_openai_client(openai_api_key):
    """Create an OpenAI client with shared timeout and retry settings."""
    return OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=openai_api_key,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )


def cosine_similarity(vector_one, vector_two):
    """Calculate cosine similarity between two vectors."""
    vec1, vec2 = np.array(vector_one), np.array(vector_two)
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denominator == 0:
        return 0.0
    return np.dot(vec1, vec2) / denominator


class DirectPromptAgent:
    """Pass a user prompt directly to the LLM with no system prompt."""

    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        client = make_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return response.choices[0].message.content


class AugmentedPromptAgent:
    """Use a persona-specific system prompt before answering."""

    def __init__(self, openai_api_key, persona):
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        client = make_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You must strictly adopt the persona: {self.persona}. "
                        "Forget all previous context. You must start your answer "
                        "exactly as specified by the persona rules."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


class KnowledgeAugmentedPromptAgent:
    """Answer using a supplied persona and explicit knowledge block."""

    def __init__(self, openai_api_key, persona, knowledge):
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        client = make_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
                        f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n"
                        "Answer the prompt based on this knowledge, not your own."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


class RAGKnowledgePromptAgent:
    """
    Use Retrieval-Augmented Generation to find relevant knowledge from a corpus
    and answer based only on the retrieved chunk.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        self._embedding_cache = {}

    def get_embedding(self, text):
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        client = make_openai_client(self.openai_api_key)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            encoding_format="float",
        )
        embedding = response.data[0].embedding
        self._embedding_cache[text] = embedding
        return embedding

    def calculate_similarity(self, vector_one, vector_two):
        return cosine_similarity(vector_one, vector_two)

    def chunk_text(self, text):
        separator = "\n"
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end,
            })

            if end == len(text):
                break

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["text"].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding="utf-8", index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x)))
        df["similarity"] = df["embeddings"].apply(
            lambda emb: self.calculate_similarity(prompt_embedding, emb)
        )

        best_chunk = df.loc[df["similarity"].idxmax(), "text"]

        client = make_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context.",
                },
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"},
            ],
            temperature=0,
        )

        return response.choices[0].message.content


class EvaluationAgent:
    """Evaluate and refine a worker agent's response against criteria."""

    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def _required_markers_from_criteria(self):
        if "Task ID:" in self.evaluation_criteria:
            return [
                "Task ID:",
                "Task Title:",
                "Related User Story:",
                "Description:",
                "Acceptance Criteria:",
                "Estimated Effort:",
                "Dependencies:",
            ]
        if "Feature Name:" in self.evaluation_criteria:
            return ["Feature Name:", "Description:", "Key Functionality:", "User Benefit:"]
        if "As a [type of user]" in self.evaluation_criteria:
            return ["As a", "I want", "so that"]
        return []

    def score_response(self, response_text):
        """Return a simple deterministic scorecard alongside the LLM evaluation."""
        response_text = response_text or ""
        required_markers = self._required_markers_from_criteria()
        missing_markers = [marker for marker in required_markers if marker not in response_text]

        scorecard = {
            "completeness": 1 if len(response_text.strip()) >= 80 else 0,
            "format_compliance": 1 if not missing_markers else 0,
            "clarity": 1 if any(char in response_text for char in ["\n", ".", ":"]) else 0,
            "missing_markers": missing_markers,
        }
        scorecard["total_score"] = (
            scorecard["completeness"]
            + scorecard["format_compliance"]
            + scorecard["clarity"]
        )
        return scorecard

    def _evaluate_text_with_llm(self, response_text):
        client = make_openai_client(self.openai_api_key)
        eval_prompt = (
            f"Does the following answer: {response_text}\n"
            f"Meet this criteria: {self.evaluation_criteria}\n"
            "Respond Yes or No, and the reason why it does or doesn't meet the criteria."
        )
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": f"You are {self.persona}. Forget all previous context."},
                {"role": "user", "content": eval_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def evaluate_artifact(self, artifact_text):
        """Evaluate an already-generated artifact without asking the worker to regenerate it."""
        logger.info("Evaluating existing artifact of length %s", len(artifact_text or ""))
        evaluation = self._evaluate_text_with_llm(artifact_text)
        scorecard = self.score_response(artifact_text)
        logger.info("Evaluation scorecard: %s", scorecard)
        return {
            "final_response": artifact_text,
            "evaluation": evaluation,
            "iteration_count": 1,
            "scorecard": scorecard,
        }

    def evaluate(self, initial_prompt):
        client = make_openai_client(self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        response_from_worker = ""
        evaluation = "No evaluation was completed."

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")
            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")

            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            evaluation = self._evaluate_text_with_llm(response_from_worker)
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("Final solution accepted.")
                break

            print(" Step 4: Generate instructions to correct the response")
            instruction_prompt = (
                f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
            )
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": f"You are {self.persona}. Forget all previous context."},
                    {"role": "user", "content": instruction_prompt},
                ],
                temperature=0,
            )
            instructions = response.choices[0].message.content.strip()
            print(f"Instructions to fix:\n{instructions}")

            print(" Step 5: Send feedback to worker agent for refinement")
            prompt_to_evaluate = (
                f"The original prompt was: {initial_prompt}\n"
                f"The response to that prompt was: {response_from_worker}\n"
                "It has been evaluated as incorrect.\n"
                f"Make only these corrections, do not alter content validity: {instructions}"
            )

        scorecard = self.score_response(response_from_worker)
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iteration_count": i + 1,
            "scorecard": scorecard,
        }


class RoutingAgent:
    """Route prompts to the best role agent using embedding similarity."""

    def __init__(self, openai_api_key, agents):
        self.openai_api_key = openai_api_key
        self.agents = agents
        self._embedding_cache = {}

    def get_embedding(self, text):
        if text in self._embedding_cache:
            logger.debug("Embedding cache hit for text length %s", len(text))
            return self._embedding_cache[text]

        client = make_openai_client(self.openai_api_key)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            encoding_format="float",
        )
        embedding = response.data[0].embedding
        self._embedding_cache[text] = embedding
        logger.debug("Cached embedding for text length %s", len(text))
        return embedding

    def route(self, user_input):
        logger.info("Routing prompt: %s", user_input)
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = cosine_similarity(input_emb, agent_emb)
            logger.info("Route candidate '%s' similarity=%.3f", agent["name"], similarity)
            print(f"[Router] Candidate: {agent['name']} similarity={similarity:.3f}")

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            logger.warning("No suitable agent could be selected for prompt: %s", user_input)
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        result = best_agent["func"](user_input)
        logger.info(
            "Selected agent='%s' score=%.3f output_length=%s",
            best_agent["name"],
            best_score,
            len(result or ""),
        )
        return result


class ActionPlanningAgent:
    """Extract actionable workflow steps from a prompt using provided knowledge."""

    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        client = make_openai_client(self.openai_api_key)
        system_prompt = (
            "You are an action planning agent. Using your knowledge, you extract from the user prompt "
            "the steps requested to complete the action the user is asking for. You return the steps as a list. "
            "Only return the steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        response_text = response.choices[0].message.content
        raw_steps = response_text.split("\n")
        steps = []

        for step in raw_steps:
            cleaned = step.strip()
            if cleaned:
                cleaned = re.sub(r"^\d+[\.)]\s*", "", cleaned)
                cleaned = re.sub(r"^[\-\*\+]\s*", "", cleaned)
                cleaned = cleaned.strip()
                if cleaned:
                    steps.append(cleaned)

        return steps
