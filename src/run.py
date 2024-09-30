import os
import argparse
import json
import jsonlines
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

from prompts import (
    DirectAsk,
    WriteThenAsk,
    ExecuteThenAsk,
    ExecuteThenAskVerbalized,
    generate_schema_prompt,
    generate_comment_prompt
)
from sql_execution import execute_model
from llms import get_llm, LLM
from utils import parse_sql

def get_ask_hint_fn(method: str, agent: LLM, database_path: str):
    if method == "DA":
        return DirectAsk(agent, database_path)
    elif method == "WA":
        return WriteThenAsk(agent, database_path)
    elif method == "EA":
        return ExecuteThenAsk(agent, database_path)
    elif method == 'EAV':
        return ExecuteThenAskVerbalized(agent, database_path)
    raise NotImplementedError

def get_exp_name(args: argparse.Namespace) -> str:
    return "__".join([
        args.series,
        args.model_name.split('/')[-1],
    ])

def setup_args() -> argparse.Namespace:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--dataset_path',
        type=str,
        default='appier-ai-research/StreamBench'
    )
    args_parser.add_argument(
        '--dataset_name',
        type=str,
        default='bird'
    )
    args_parser.add_argument(
        '--split',
        type=str,
        default='test'
    )
    args_parser.add_argument(
        '--db_root_path',
        type=str,
        default="./data/bird/dev_databases"
    )
    args_parser.add_argument(
        '--series',
        type=str,
        required=True,
        choices=["openai", "hf_model"]
    )
    args_parser.add_argument(
        '--model_name',
        type=str,
        required=True,
    )
    args_parser.add_argument(
        '--exp_name',
        type=str,
        default=None
    )
    args_parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=["DA", "WA", "EA", "EAV"]
    )  # DA: direct ask; WA: write then ask; EA: execute then ask, EAV: execute then ask (verbalized)
    args_parser.add_argument(
        "--output_dir",
        type=str,
        default="./results"
    )
    return args_parser.parse_args()

if __name__ == "__main__":
    args = setup_args()
    if args.exp_name is not None:
        exp_name = args.exp_name
    else:
        exp_name = get_exp_name(args)
    print(f"Running experiment: {exp_name}")
    dataset = load_dataset(args.dataset_path, args.dataset_name)[args.split]
    print("Dataset loaded.")
    # Get prompt schemas
    db_prompt_schema = {}
    for row in tqdm(dataset, desc="Generating DB schemas"):
        if row['db_id'] in db_prompt_schema:
            continue
        db_path = os.path.join(args.db_root_path, row['db_id'], row['db_id'] + '.sqlite')
        db_prompt_schema[row['db_id']] = generate_schema_prompt(db_path)
    # Setup the agent & ask_hint_functions
    agent = get_llm(args.series, args.model_name)
    ask_hint_fn = get_ask_hint_fn(args.method, agent, args.db_root_path)

    # Skip runned_time step
    cached_rows = list()
    runned_time_steps = 0
    result_output = os.path.join(args.output_dir, f"{exp_name}.jsonl")
    Path(result_output).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(result_output):
        with jsonlines.open(result_output) as reader:
            cached_rows = list(reader)
            runned_time_steps = len(cached_rows)

    for time_step, row in enumerate(tqdm(dataset, dynamic_ncols=True)):
        if runned_time_steps > time_step:  # Skip runned_time step
            continue
        db_path = os.path.join(args.db_root_path, row['db_id'], row['db_id'] + '.sqlite')
        schema_prompt = db_prompt_schema[row['db_id']]
        question = row['question']

        if cached_rows:
            cached_row = cached_rows[time_step]
            
            gen_sql = cached_row["gen_sql"]
            gen_sql_w_hint = cached_row["gen_sql_w_hint"]
            
            correct = cached_row["correct"]
            correct_w_hint = cached_row["correct_w_hint"]
        else:
            # Original generated SQL
            question_prompt = generate_comment_prompt(question, knowledge=None)  # NOTE: no retrieved knowledge in this no streaming setting
            combined_prompts = schema_prompt + '\n\n' + question_prompt + 'Now, generate the correct SQL code directly in the format of ```sql\n<your_SQL_code>\n```:'
            response, _ = agent(combined_prompts)
            gen_sql = parse_sql(response)
            
            # Generated SQL with hint
            question_prompt_w_hint = generate_comment_prompt(question, knowledge=row["evidence"])
            combined_prompts_w_hint = schema_prompt + '\n\n' + question_prompt_w_hint + 'Now, generate the correct SQL code directly in the format of ```sql\n<your_SQL_code>\n```:'
            response_w_hint, _ = agent(combined_prompts_w_hint)
            gen_sql_w_hint = parse_sql(response_w_hint)

            # Evaluation
            correct, (pred, gt) = execute_model(gen_sql, row['SQL'], db_path)
            correct_w_hint, (pred_w_hint, gt) = execute_model(gen_sql_w_hint, row['SQL'], db_path)
        
        # method2hint_res = dict()
        # NOTE: hint_res should include output logits (token-logit pairs) if available
        hint_res = ask_hint_fn(
            db_schema=schema_prompt,
            question=question,
            gen_sql=gen_sql,
            db_id=row["db_id"]
        )
        # method2hint_res[method] = hint_res
        
        # Logging
        output = dict()
        output["correct"] = correct
        output["correct_w_hint"] = correct_w_hint
        output["time_step"] = time_step
        for key, value in row.items():
            output[key] = value
        # for method, hint_res in method2hint_res.items():
        output[args.method] = dict()
        for key, value in hint_res.items():
            output[args.method][key] = value
        output["model_name"] = args.model_name
        output["gen_sql"] = gen_sql
        output["gen_sql_w_hint"] = gen_sql_w_hint
        
        with open(result_output, 'a') as fout:
            fout.write(json.dumps(output) + '\n')
