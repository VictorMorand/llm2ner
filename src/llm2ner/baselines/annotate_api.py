import os, re, requests, subprocess, time, socket, json
from pathlib import Path
import logging
from tqdm import tqdm
from llm2ner.data import load_dataset_splits
from llm2ner.utils import PathOutput
from transformers import AutoTokenizer
from experimaestro import Task, Param, Meta, Constant, field, PathGenerator

# this is needed to avoid proxy issues when lauching ollama locally
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

#### OLLMA SERVER MANAGEMENT ####

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

PROMPT = (
    "Given a passage, extract ALL entities. Retrieve ANY possible spans including simple, nested, overlapping and unspecific.\n"
    "Format your answer as:\n\"\n- entity1\n- entity2\n- ...\n\"\n"
    "\nPassage:\n\"\n{sentence}\n\"\n"
)


def is_ollama_running(host="localhost", port=11434):
    """Check if Ollama server is running on the given host and port."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False


def launch_ollama_server(model: str = "llama3.2"):
    """Launch Ollama server as a subprocess if not already running."""
    host = "localhost"
    port = 11434
    if is_ollama_running(host, port):
        logging.info(f"Ollama server already running at {host}:{port}")
        return None
    logging.info("Starting Ollama server...")
    log_file = open("ollama.log", "a")
    # Start ollama serve in background
    proc = subprocess.Popen(["ollama", "serve"], stdout=log_file, stderr=log_file)
    # Wait for server to be ready
    for _ in range(30):
        if is_ollama_running(host, port):
            logging.info("Ollama server started.")
            break
        time.sleep(1)
    else:
        log_file.close()
        raise RuntimeError("Ollama server did not start in time.")

    # Ensure the model is loaded by running 'ollama run <model>'
    logging.info(f"Ensuring Ollama model '{model}' is loaded...")
    try:
        subprocess.run(
            ["ollama", "run", model], check=True, stdout=log_file, stderr=log_file
        )
        print(f"Model '{model}' loaded.")
    except Exception as e:
        print(f"Failed to load model '{model}': {e}")
        proc.terminate()
        proc.wait()
        log_file.close()
        raise
    return proc


def ollama_infer(prompt, model, url):
    endpoint = f"{url}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"Ollama inference error: {e}")
        return ""


def openai_infer(prompt, model, client):
    try:
        # prompt = prompt.split("Passage:")
        # system = prompt[0]
        # message = ''.join(prompt[1:]).strip()
    
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI inference error: {e}")
        return ""


#### PROMPTING & PARSING ####


def format_prompt(sentence):
    return PROMPT.format(sentence=sentence)


def parse_entities(llm_output: str):
    pattern = r"^-+\s*(.+)$"
    llm_output = llm_output.replace("\n*", "\n-")
    entities = re.findall(pattern, llm_output, flags=re.MULTILINE)
    return [e.strip() for e in entities if e.strip()]


def find_occurrences(sentence, search_string):
    # Collapse multiple spaces in search string
    normalized_search = re.sub(r"\s+", " ", search_string).strip()

    # Escape regex special characters
    escaped_search = re.escape(normalized_search)

    # Make spaces flexible: allow any whitespace (including none) around punctuation
    # Example: "IL-2 )" -> "IL-2\s*\)"
    # This ensures ( IL-2 ) matches (IL-2) as well
    pattern = re.sub(r"\\\s+", r"\\s*", escaped_search)
    # Add word boundaries to avoid partial matches
    pattern = rf"\b{pattern}\b"

    matches = re.finditer(pattern, sentence, flags=re.IGNORECASE)
    return [(match.start(), match.end(), match.group()) for match in matches]


def get_items(text, entity) -> list:
    # Escape special regex chars but keep spaces normal
    pattern = re.escape(entity)
    # Find all matches
    return [
        {
            "entity": matched_text,
            "entity_pos": (start, end),
        }
        for start, end, matched_text in find_occurrences(text, entity)
    ]


class LLMannotation(Task):

    llm_name: Param[str]
    """Name of the LLM model to use (as known by the LLM server)"""
    data_name: Param[str] = "MultiNERD"
    """Name of the dataset to use (CoNLL 2003, OntoNotes 5, WNUT 17, etc.)"""
    max_length: Param[int]
    """Maximum sequence length"""
    max_ent_length: Param[int]
    """Maximum entity length in chars"""
    limit_samples: Param[int] = -1
    """Limit the number of samples to process, -1 for no limit"""

    # Meta Params
    version: Constant[str] = "1.2"
    """Version of the task definition"""
    data_folder: Meta[str] = ""
    """Path to the folder containing the datasets"""
    api_url: Meta[str] = ""
    """URL of the LLM API server"""

    def task_outputs(self, dep):
        return dep(PathOutput.C(path=Path(self.jobpath)))

    def res_folder_name(self):
        llm_name = self.llm_name.replace(":", "_").replace("/", "_").replace(".", "_")
        name = f"{self.data_name}_{llm_name}"
        if self.limit_samples > 0:
            name += f"_n{self.limit_samples}"
        return name

    def execute(self):
        # Ensure Ollama server is running before making API calls

        if self.llm_name.lower().startswith("gpt") and not self.llm_name.startswith("gpt-oss"):
            import openai
            openai.api_key = os.environ.get("OPENAI_API_KEY", "")
            assert openai.api_key != "", "OPENAI_API_KEY environment variable not set"

            logging.info("Using OpenAI API for LLM calls.")
            ollama_proc = None
            client = openai.OpenAI()
            use_ollama = False
        else:
            logging.info("Using Ollama API for LLM calls.")
            ollama_proc = launch_ollama_server(model=self.llm_name)
            client = None
            use_ollama = True

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        dataset = load_dataset_splits(
            dataset_name=self.data_name,
            data_folder=self.data_folder,
            tokenizer=tokenizer,
            max_length=self.max_length,
            max_ent_length=self.max_ent_length,
            mode="last",
        )
        if self.limit_samples > 0:
            logging.info(f"Limiting to maximum {self.limit_samples} samples per split")
            for split in dataset:
                limit = min(self.limit_samples, len(dataset[split]))
                dataset[split].data = dataset[split].data[: limit]

        for split in dataset:
            print(f"Dataset split '{split}' has {len(dataset[split])} samples.")
            annotated_data = []
            for item in tqdm(dataset[split], desc=f"Annotating {split}"):
                sent = item["text"]
                prompt = format_prompt(sent)
                if use_ollama:
                    llm_output = ollama_infer(
                        prompt, model=self.llm_name, url=self.api_url
                    )
                else:
                    llm_output = openai_infer(
                        prompt, model=self.llm_name, client=client
                    )

                entities = parse_entities(llm_output)

                # Example: find spans for all entities
                entity_spans = []
                for ent in entities:
                    entity_spans += get_items(sent, ent)

                annotated_data.append(
                    {
                        "sentence": sent,
                        "entities": [
                            {
                                "name": e["entity"],
                                "type": "Entity",
                                "pos": list(e["entity_pos"]),
                            }
                            for e in entity_spans
                        ],
                        "llm_output": llm_output,
                    }
                )
            # write results to file
            result_folder = Path(self.res_folder_name())
            result_folder.mkdir(parents=True, exist_ok=True)
            result_file = result_folder / f"{split}.json"
            with open(result_file, "w") as f:
                json.dump(annotated_data, f, indent=2)
            print(f"Wrote {len(annotated_data)} annotated samples to {result_file}")
            # write last prompt to file for debugging
            debug_file = result_folder / f"last_prompt.txt"
            with open(debug_file, "w") as f:
                f.write(prompt)

        if ollama_proc is not None:
            print("Terminating Ollama server...")
            ollama_proc.terminate()
            ollama_proc.wait()
            print("Ollama server terminated.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    USER = os.environ.get("USER", "unknown_user")
    data_folder = (
        "/Users/victor/code/data/NER" if USER == "victor" else "/data/morand/NER"
    )
    task = LLMannotation.C(
        llm_name="gpt-4.1", # "gpt-3.5-turbo" gpt-4o-mini llama3:8b gemma3
        data_name="MultiNERD",
        data_folder=data_folder,
        api_url=OLLAMA_URL,
        max_length=1200,
        max_ent_length=500,
        limit_samples=1000,
    )
    task.execute()
