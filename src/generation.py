"""
This file contains all helper functions to the generation step in the pipeline.
It includes:
- get_api_client
- run_llm
"""
import os
import json
from pathlib import Path
from openai import OpenAI
import time
import uuid
import google.generativeai as genai
import re
from google.generativeai.types import HarmCategory, HarmBlockThreshold


import yaml
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"  

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def get_api_client(model_name):
    """
    Create and return an API client for the specified provider.

    Params:
    - model_name:
        - "gpt"
        - "deepseek"
        - "qwen"
        - "gemini"

    Returns:
    - client: OpenAI client instance (for gpt/deepseek), or configured genai module (for gemini)

    """
    model_name = model_name.lower()

    if model_name == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment.")
        return OpenAI(api_key=api_key)

    elif model_name == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not found in environment.")
        return OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    elif model_name == "qwen":
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise RuntimeError("HF_TOKEN not found in environment.")
        return OpenAI(
            api_key=api_key,
            base_url="https://router.huggingface.co/v1"
        )
    
    elif model_name == "llama":                                                                                                                                
        return OpenAI(         
          api_key="vllm",                                                                                                                                    
          base_url="http://localhost:8000/v1"
      )       
    
    
    elif model_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment.")
        genai.configure(api_key=api_key)
        return genai  # return the configured module itself as the "client"

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def _estimate_cost( 
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
):
    """
    Estimate generation cost in USD. GPT generated (#TODO: check if pricing is accurate)

    Parameters
    ----------
    model_name : str
        Supported values:
        - "gpt"
        - "deepseek"
    prompt_tokens : int
    completion_tokens : int

    Returns
    -------
    float
        Estimated cost in USD
    """
    # Prices per 1K tokens (USD)
    PRICES_PER_1K = {
        "gpt": {
            "prompt": 0.00015,      # $0.15 / 1M tokens
            "completion": 0.00060,  # $0.60 / 1M tokens
        },
        "deepseek": {
            "prompt": 0.00010,      # conservative estimate
            "completion": 0.00020,
        },
        "gemini": {
        "prompt": 0.00030,      # $0.30 / 1M tokens (gemini-2.5-flash)
        "completion": 0.00250, 
        },
        "qwen": {
        "prompt": 0.00012,      # $0.12 / 1M tokens (Qwen2.5-72B-Instruct)
        "completion": 0.00039,  # $0.39 / 1M tokens
        }
    }

    if model_name not in PRICES_PER_1K:
        raise ValueError(f"Unknown model_name for cost estimation: {model_name}")

    prices = PRICES_PER_1K[model_name]

    cost = (
        (prompt_tokens / 1000) * prices["prompt"]
        + (completion_tokens / 1000) * prices["completion"]
    )

    return round(cost, 6)


def run_llm(client,
            model_name: str, 
            news_article: dict,
            prompt: dict,
            max_tokens: int,
            rep_index: int):
    """
    Generates an output given model name, a news article and a prompt
    
    Params:
    - client: initialized API connection using get_api_client()
    - model_name: name of the model
        - "gpt" for GPT 4-o mini
        - "deepseek" for DeepSeek
        - "gemini" for Gemini
        - "qwen" for Qwen
    - news_article: dictionary containing information about news article
    - prompt: dictionary containing information about prompt
    - max_tokens: maximum number of tokens for the output

    Returns:
    - records: DF containing generated response and estimated time and costs
    """
    # -----------------------------
    # Prompt set up
    # -----------------------------
    articleID = news_article["articleID"]
    news_title = news_article["title"]
    news_headline = news_article["headline"]

    promptID = prompt["promptID"]
    user_prompt = prompt["user_prompt"]
    sys_prompt = prompt["system_prompt"]

    if prompt["task_type"] == "tweet_generation_dual_viewpoint":
        max_tokens = 2048 if model_name == "gemini" else 300 # reason:

    # -----------------------------
    # LLM set up
    # -----------------------------
    model_config = None

    for llm_cfg in config["LLMs"]:
        if llm_cfg["model_name"] == model_name:
            model_config = llm_cfg
            break

    if model_config is None:
        raise ValueError(f"Unsupported model name {model_name}")

    model_id = model_config["model_id"]

    # -----------------------------
    # Generation
    # -----------------------------
    start_time = time.time()
    if model_name in("gpt", "deepseek", "qwen", "llama"):
        result = _run_chat_completion_openai( client = client,
                                        model_id = model_id,
                                        news_title = news_title,
                                        news_headline = news_headline, 
                                        user_prompt = user_prompt,
                                        sys_prompt = sys_prompt,
                                        max_tokens = max_tokens)
    elif model_name == "gemini":
        result = _run_chat_completion_gemini( client = client,
                                        model_id = model_id,
                                        news_title = news_title,
                                        news_headline = news_headline, 
                                        user_prompt = user_prompt,
                                        sys_prompt = sys_prompt,
                                        max_tokens = max_tokens

        )
    

    end_time = time.time()

    generated_text = result["text"]
    generation_time = end_time - start_time
    
    # -----------------------------
    # Storing results
    # -----------------------------
    if result["usage"] is not None:
        generation_cost = _estimate_cost(model_name = model_name, 
                                      prompt_tokens = result["usage"]["prompt_tokens"],
                                      completion_tokens = result["usage"]["completion_tokens"]) 
    else:
        generation_cost = None

    records = []
    if prompt["task_type"] == "tweet_generation_dual_viewpoint":
        try:
            obj = json.loads(generated_text.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed at position {e.pos}")
            print(f"Character at error: {repr(generated_text[e.pos-5:e.pos+5])}")
            raise
        # obj = json.loads(generated_text)
        records = [
            {
            'genID': str(uuid.uuid4()), # identifier for this generation
            'rep_index': rep_index,
            'articleID': articleID, 
            'modelID': model_name, 
            'promptID': promptID,
            "viewpoint": obj.get("viewpoint_1"),
            'generated_text': obj.get("tweet_1"), 
            'generation_time': generation_time/2,
            'generation_cost': generation_cost/2 if generation_cost is not None else None
        },
        {
            'genID': str(uuid.uuid4()), # identifier for this generation
            'rep_index': rep_index,
            'articleID': articleID, 
            'modelID': model_name, 
            'promptID': promptID,
            "viewpoint": obj.get("viewpoint_2"),
            'generated_text': obj.get("tweet_2"), 
            'generation_time': generation_time/2,
            'generation_cost': generation_cost/2 if generation_cost is not None else None
        }
        ]
    else:
        records = {
            'genID': str(uuid.uuid4()), # identifier for this generation
            'rep_index': rep_index,
            'articleID': articleID, 
            'modelID': model_name, 
            'promptID': promptID,
            'generated_text': generated_text, 
            'generation_time': generation_time,
            'generation_cost': generation_cost
        }

    _append_generation(records,
                      base_dir = "data/generations",
                      model_name = model_name)
    return records 

   

def _run_chat_completion_openai(
    client: OpenAI,
    model_id: str,
    news_title: str,
    news_headline: str,
    user_prompt: str,
    sys_prompt: str | None,
    max_tokens: int,
    temperature: float = 0.7,
):
    filled_user_prompt = (
        user_prompt.replace("{{article_title}}", news_title)
                  .replace("{{article_headline}}", news_headline)
    )

    messages = []
    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": filled_user_prompt})

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    text = response.choices[0].message.content

    usage = response.usage
    usage_dict = None
    if usage is not None:
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    return {
        "text": text,
        "usage": usage_dict,
        "model": getattr(response, "model", model_id)
    }
   
def _run_chat_completion_gemini(
    client,
    model_id: str,
    news_title: str,
    news_headline: str,
    user_prompt: str,
    sys_prompt: str | None,
    max_tokens: int,
    temperature: float = 0.7,
):
    filled_user_prompt = (
        user_prompt.replace("{{article_title}}", news_title)
                   .replace("{{article_headline}}", news_headline)
    )

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model = client.GenerativeModel(
        model_name=model_id,
        system_instruction=sys_prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    try:
        response = model.generate_content(filled_user_prompt)

        if not response.candidates or not response.candidates[0].content.parts:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "EMPTY_RESPONSE"
            feedback = getattr(response, "prompt_feedback", "No specific feedback")
            raise RuntimeError(f"Gemini generation blocked. Reason: {finish_reason}. Feedback: {feedback}")

        text = response.candidates[0].content.parts[0].text

        # Strip markdown code fences if present
        if text.strip().startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text.strip())
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        usage_dict = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage_dict = {
                "prompt_tokens": um.prompt_token_count,
                "completion_tokens": um.candidates_token_count,
                "total_tokens": um.total_token_count,
            }

        return {
            "text": text,
            "usage": usage_dict,
            "model": model_id,
        }

    except RuntimeError:
        raise  # re-raise our own errors as-is
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed unexpectedly: {e}") from e
    

def _append_generation(
    records: dict,
    base_dir: Path | str,
    model_name: str
):
    """
    Append one or more generation records to a JSONL file.

    Each line in the file is one JSON object.
    """
    dataset_nr = config["Run"]["dataset_nr"]

    base_dir = Path(base_dir) 
    out_dir = base_dir / f"dataset_{dataset_nr}"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / f"generations_{model_name}.jsonl"

    # Normalize to list
    if not isinstance(records, list):
        records = [records]

    with file_path.open("a", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


