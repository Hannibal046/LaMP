"""
Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
"""

# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass  # for storing API inputs, outputs, and metadata
from tqdm import tqdm
from utils import (
    get_bleu_score,
    get_chrfpp_score,
    get_rouge_score,
    get_ter_score,
)

def eval_generation(hyps,refs,trg_lang):
    return {
        "bleu":get_bleu_score(hyps,refs,trg_lang=trg_lang),
        "chrf++":get_chrfpp_score(hyps,refs),
        "rouge":get_rouge_score(hyps,refs),
        # "ter":get_ter_score(hyps,refs,trg_lang=trg_lang),
    }

Code2Lang = {
    "en":"English",
    "de":"German",
    "es":"Spanish",
    "zh":"Chinese",
}

async def process_api_requests_from_file(
    input_data,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
    openai_output,
    prog_bar,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""

    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    requests = iter(input_data)
    logging.debug(f"File opened. Entering main loop")

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
            elif file_not_finished:
                try:
                    # get new request
                    metadata_and_modelinput = next(requests)
                    meta_data,request_json = metadata_and_modelinput['meta_data'],metadata_and_modelinput['openai_input']
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request_json=request_json,
                        meta_data = meta_data,
                        token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
                        attempts_left=max_attempts,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.debug("Read file exhausted")
                    file_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.token_consumption
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        request_url=request_url,
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        save_filepath=save_filepath,
                        status_tracker=status_tracker,
                        output = openai_output,
                        prog_bar = prog_bar,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

    # after finishing, log final status
    logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")


# dataclasses
@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    meta_data: dict
    token_consumption: int
    attempts_left: int
    result = []


    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
        output: list,
        prog_bar,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                await asyncio.sleep(2)
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                # append_to_jsonl([self.request_json, self.result], save_filepath)
                output.append(
                    {
                        "meta_data":self.meta_data,
                        "openai_input":self.request_json,
                        "error":[str(x) for x in self.result],
                    }
                )
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            
            # append_to_jsonl([self.request_json, response], save_filepath)
            output.append(
                {
                    "meta_data":self.meta_data,
                    "openai_input":self.request_json,
                    "response":response,
                }
            )
            prog_bar.update(1)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions
def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script





def reorder(data):
    ret = [None for x in range(len(data))]
    for d in data:
        idx = d['meta_data']["id"]
        ret[idx] = d
    return ret

async def main(
    openai_input,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
    trg_lang: str,
):
    
    openai_output = []
    prog_bar = tqdm(total=len(openai_input))
    await process_api_requests_from_file(
            input_data=openai_input,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name=token_encoding_name,
            max_attempts=int(max_attempts),
            logging_level=int(logging_level),
            openai_output = openai_output,
            prog_bar = prog_bar,
        )
    prog_bar.close()

    assert len(openai_output)==len(openai_input)
    openai_output = reorder(openai_output)

    ## calculate metrics and cost
    hyps = [x['response']['choices'][0]['message']['content'] for x in openai_output]
    refs = [x['meta_data']['ref'] for x in openai_output]
    
    eval_metrics = eval_generation(hyps,refs,trg_lang=trg_lang)
    print(json.dumps(eval_metrics,indent=4))

    total_tokens = sum(x['response']['usage']['total_tokens'] for x in openai_output)
    total_cost = total_tokens / 1000 * 0.002
    print("total tokens:",total_tokens,'\ntotal cost:',total_cost,"$")

    ## save
    if save_filepath is not None:
        with open(save_filepath,'w') as f:
            for hyp in hyps:
                f.write(hyp+'\n')
    


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath",default='data/jrc_acquis/ende/test.jsonl')
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--request_url", default="https://api.openai.com/v1/chat/completions")
    # parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--api_key", default="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")
    parser.add_argument("--max_requests_per_minute", type=int, default=2000)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=40)
    parser.add_argument("--src",default='en')
    parser.add_argument("--trg",default='de')
    parser.add_argument("--prompt",default="Translate the following <src_lang> text to <trg_lang>: ")
    args = parser.parse_args()

    print(f"Prompt:\n{args.prompt.replace('<src_lang>',Code2Lang[args.src]).replace('<trg_lang>',Code2Lang[args.trg])}")

    ## load data
    with open(args.requests_filepath) as f:
        data = [json.loads(x) for x in f.readlines()]
    
    ## prepare openai input
    openai_input = []
    for idx,d in enumerate(data):
        single_input = {}
        single_input['meta_data'] = {"id":idx,"ref":d[args.trg],}
        
        content = args.prompt.replace("<src_lang>",Code2Lang[args.src]).replace("<trg_lang>",Code2Lang[args.trg]) + d[args.src]
        single_input['openai_input'] = {
                        "model":"gpt-3.5-turbo",
                        "messages":[
                                    {"role": "user", "content": content}
                                    ]
                        }
        
        openai_input.append(single_input)

    # run script
    asyncio.run(main(
            openai_input=openai_input,
            save_filepath=args.save_filepath,
            request_url=args.request_url,
            api_key=args.api_key,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_tokens_per_minute=float(args.max_tokens_per_minute),
            token_encoding_name=args.token_encoding_name,
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
            trg_lang=args.trg,
        )
    )