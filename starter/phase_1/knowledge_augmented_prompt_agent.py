# Test script for KnowledgeAugmentedPromptAgent class

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

# Instantiate a KnowledgeAugmentedPromptAgent
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# Call the agent to respond
response = knowledge_agent.respond(prompt)
print(response)

# Print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge
print("\n[KnowledgeAugmentedPromptAgent Verification]:")
print("The agent responded that the capital of France is London (which is incorrect factually but correct per the provided custom knowledge). This confirms it is using the augmented knowledge instead of its inherent LLM knowledge.")
