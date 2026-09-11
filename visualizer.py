import math
import networkx as nx
import openai
import config
from typing import List, Dict

class TokenPredictor:
    def __init__(self, model_name: str = config.OPENAI_MODEL):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def predict_tokens(self, prompt: str, max_tokens: int = 1200):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )

        predictions = []
        content_text = response.choices[0].message.content or ""
        logprobs_content = response.choices[0].logprobs.content if response.choices[0].logprobs else []

        for token_info in logprobs_content:
            token = token_info.token
            chosen_logprob = token_info.logprob
            prob = math.exp(chosen_logprob)

            top_candidates = []
            if token_info.top_logprobs:
                for top_item in token_info.top_logprobs:
                    top_candidates.append({
                        "token": top_item.token,
                        "probability": math.exp(top_item.logprob),
                    })

            predictions.append({
                "token": token,
                "probability": prob,
                "top_candidates": top_candidates,
            })

        # Return both full text string and predictions list
        return content_text, predictions


def create_token_graph(model_name: str, predictions: List[Dict]) -> nx.DiGraph:
    """Builds a NetworkX directed graph representing token predictions."""
    G = nx.DiGraph()
    G.add_node("START", label="START", type="root")
    
    prev_node = "START"
    for idx, pred in enumerate(predictions):
        curr_node = f"step_{idx}_{pred['token']}"
        G.add_node(curr_node, label=pred["token"], prob=pred["probability"])
        G.add_edge(prev_node, curr_node, weight=pred["probability"])
        prev_node = curr_node
        
    return G