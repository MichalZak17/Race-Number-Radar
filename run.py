import io
import re
import os
import json
import shutil
import base64
import typer
import concurrent.futures

from PIL import Image
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from collections import defaultdict
from openai import OpenAI, APIError
from typing import Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

SITE_URL = "http://localhost"
SITE_NAME = "RaceNumberRadar"
DEFAULT_API_MODEL = "qwen/qwen2.5-vl-32b-instruct"
DEFAULT_WORKERS = 8
DEFAULT_MIN_BIB_LEN = 3
DEFAULT_MAX_BIB_LEN = 4
DEFAULT_PROVIDER = "deepinfra/bf16"
DEFAULT_MAX_SIZE_KB = 1500
CONFIG_FILE = "config.json"

app = typer.Typer()
console = Console()

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")


def load_config() -> dict:
    """
    Loads configuration settings from a JSON file.

    Returns:
        dict: The configuration settings loaded from the file. If the file does not exist, returns an empty dictionary.
    """
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def save_config(config: dict):
    """
    Saves the given configuration dictionary to a JSON file.

    Args:
        config (dict): The configuration data to be saved.

    Side Effects:
        Writes the configuration to the file specified by CONFIG_FILE in JSON format.
        Prints a confirmation message to the console upon successful save.
    """
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
    console.print(f"[bold green]Configuration saved to {CONFIG_FILE}[/bold green]")


def scan_directory(directory: str) -> list[str]:
    """
    Scans the specified directory for image files with supported extensions.

    Args:
        directory (str): The path to the directory to scan.

    Returns:
        list[str]: A list of file paths for image files found in the directory.
    """
    files = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for item in os.scandir(directory):
        if item.is_file() and os.path.splitext(item.name)[1].lower() in supported_extensions:
            files.append(item.path)
    return files


def image_to_base64_uri(
    image: Image.Image, max_size_kb: int = DEFAULT_MAX_SIZE_KB
) -> str:
    """
    Converts a PIL Image to a base64-encoded JPEG URI, ensuring the encoded image does not exceed a specified size.

    Args:
        image (Image.Image): The PIL Image to encode.
        max_size_kb (int, optional): The maximum allowed size of the encoded image in kilobytes. Defaults to DEFAULT_MAX_SIZE_KB.

    Returns:
        str: A base64-encoded JPEG image URI suitable for embedding in HTML.

    Notes:
        The function reduces JPEG quality in steps of 5 until the encoded image fits within the specified size or the quality reaches 10.
    """
    quality = 95
    while True:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        if buffered.tell() / 1024 <= max_size_kb or quality <= 10:
            break
        quality -= 5
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(APIError),
)
def process_single_image(
    file_path: str, client: OpenAI, min_bib_len: int, max_bib_len: int
) -> tuple[str, list[str]]:
    """
    Processes a single image to identify race bib numbers using an AI model.

    Args:
        file_path (str): Path to the image file to be processed.
        client (OpenAI): An OpenAI client instance for making API requests.
        min_bib_len (int): Minimum length of bib numbers to detect.
        max_bib_len (int): Maximum length of bib numbers to detect.

    Returns:
        tuple[str, list[str]]: A tuple containing the file path and a list of detected bib numbers as strings.
            If no bib numbers are found or an error occurs, the list will be empty.

    Raises:
        APIError: If an API error occurs during processing (retries up to 5 times).
        Exception: For other errors (e.g., corrupted image files), logs the error and returns an empty list.
    """
    try:
        with Image.open(file_path).convert("RGB") as image:
            image.thumbnail((1024, 1024))
            base64_image_uri = image_to_base64_uri(image)

        prompt_text = (
            f"Identify all race bib numbers in this image. "
            f"Respond with only the numbers, separated by commas. "
            f"For example: 123,431,890. The bib numbers are always between {min_bib_len} and {max_bib_len} digits. If no numbers are found, respond with 'none'."
        )

        extra_body_params = {"provider": {"only": [DEFAULT_PROVIDER]}}

        completion = client.chat.completions.create(
            extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_NAME},
            model=DEFAULT_API_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": base64_image_uri}},
                    ],
                }
            ],
            max_tokens=30,
            extra_body=extra_body_params,
        )

        response_text = completion.choices[0].message.content.strip().lower()

        if "none" in response_text or not response_text:
            return file_path, []

        detected_numbers = re.findall(r"\d+", response_text)
        filtered_numbers = [
            num for num in detected_numbers if min_bib_len <= len(num) <= max_bib_len
        ]
        return file_path, filtered_numbers

    # Catch non-API errors (e.g., corrupted image files) that tenacity won't retry.
    except Exception as e:
        console.print(f"[bold red]Error processing {os.path.basename(file_path)}:[/bold red] {e}")
        return file_path, []


@app.command()
def process(
    directory: str = typer.Argument(..., help="Directory with images to process."),
    api_model: Optional[str] = typer.Option(
        None, help="API model to use (overrides config)."
    ),
    workers: Optional[int] = typer.Option(
        None, help="Number of workers (overrides config)."
    ),
    min_bib_len: Optional[int] = typer.Option(
        None, help="Minimum bib number length (overrides config)."
    ),
    max_bib_len: Optional[int] = typer.Option(
        None, help="Maximum bib number length (overrides config)."
    ),
    max_size_kb: Optional[int] = typer.Option(
        None, help="Max image size in KB for base64 encoding (overrides config)."
    ),
):
    """
    Processes a directory of images to detect race numbers and organizes them into subdirectories by detected number.

    Args:
        directory (str): Directory containing images to process.
        api_model (Optional[str], optional): API model to use for detection (overrides config).
        workers (Optional[int], optional): Number of worker threads for parallel processing (overrides config).
        min_bib_len (Optional[int], optional): Minimum bib number length to detect (overrides config).
        max_bib_len (Optional[int], optional): Maximum bib number length to detect (overrides config).
        max_size_kb (Optional[int], optional): Maximum image size in KB for base64 encoding (overrides config).

    Workflow:
        - Loads configuration and applies overrides from CLI arguments.
        - Scans the specified directory for image files.
        - Initializes the OpenAI API client.
        - Processes images in parallel, detecting race numbers in each image.
        - Organizes images into subdirectories named after detected race numbers.
        - Provides progress updates and error reporting.

    Returns:
        None
    """
    config = load_config()

    effective_api_model = api_model or config.get("api_model", DEFAULT_API_MODEL)
    effective_workers = workers or config.get("workers", DEFAULT_WORKERS)
    effective_min_bib_len = min_bib_len or config.get(
        "min_bib_len", DEFAULT_MIN_BIB_LEN
    )
    effective_max_bib_len = max_bib_len or config.get(
        "max_bib_len", DEFAULT_MAX_BIB_LEN
    )
    effective_max_size_kb = max_size_kb or config.get(
        "max_size_kb", DEFAULT_MAX_SIZE_KB
    )

    files = scan_directory(directory)
    if not files:
        console.print(f"[bold red]No image files found in {directory}.[/bold red]")
        return

    console.print(
        f"Processing [bold blue]{len(files)}[/bold blue] files in [bold yellow]{directory}[/bold yellow] with {effective_workers} workers..."
    )
    console.print(f"Using model: [bold cyan]{effective_api_model}[/bold cyan]")

    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize API client: {e}[/bold red]")
        return

    number_to_images = defaultdict(list)

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(files))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_workers
        ) as executor:
            future_to_file = {
                executor.submit(
                    process_single_image,
                    file_path,
                    client,
                    effective_min_bib_len,
                    effective_max_bib_len,
                ): file_path
                for file_path in files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    file_path, detected_numbers = future.result()
                    if detected_numbers:
                        for number in detected_numbers:
                            number_to_images[number].append(file_path)
                        progress.console.print(
                            f"File: [green]{os.path.basename(file_path)}[/green] -> Detected: [bold cyan]{', '.join(detected_numbers)}[/bold cyan]"
                        )
                except Exception as e:
                    file_path = future_to_file[future]
                    console.print(
                        f"[bold red]A task failed for {os.path.basename(file_path)} after all retries: {e}[/bold red]"
                    )

                progress.update(task, advance=1)

    console.print("\n--- Organizing files ---")
    if not number_to_images:
        console.print("[yellow]No valid numbers were detected to organize.[/yellow]")
        return

    for number, images in number_to_images.items():
        number_dir = os.path.join(directory, number)
        os.makedirs(number_dir, exist_ok=True)
        for img_path in set(images):
            try:
                shutil.copy(img_path, number_dir)
            except Exception as e:
                console.print(f"[bold red]Error copying {img_path} to {number_dir}: {e}[/bold red]")
        console.print(f"Organized {len(set(images))} image(s) for number [bold magenta]{number}[/bold magenta] into [bold green]'{number_dir}'[/bold green]")


@app.command()
def set_config(
    api_model: Optional[str] = typer.Option(None, help="Set API model."),
    workers: Optional[int] = typer.Option(None, help="Set number of workers."),
    min_bib_len: Optional[int] = typer.Option(
        None, help="Set minimum bib number length."
    ),
    max_bib_len: Optional[int] = typer.Option(
        None, help="Set maximum bib number length."
    ),
    max_size_kb: Optional[int] = typer.Option(None, help="Set max image size in KB."),
):
    """
    Sets configuration values for the application and saves them to config.json.

    Parameters:
        api_model (Optional[str]): The API model to use.
        workers (Optional[int]): The number of worker processes.
        min_bib_len (Optional[int]): The minimum length of a bib number.
        max_bib_len (Optional[int]): The maximum length of a bib number.
        max_size_kb (Optional[int]): The maximum image size in kilobytes.

    If any option is provided, updates the corresponding value in the configuration file.
    If no options are provided, displays a message indicating that no changes were made.
    """
    config = load_config()
    updated = False

    if api_model is not None:
        config["api_model"] = api_model
        updated = True
    if workers is not None:
        config["workers"] = workers
        updated = True
    if min_bib_len is not None:
        config["min_bib_len"] = min_bib_len
        updated = True
    if max_bib_len is not None:
        config["max_bib_len"] = max_bib_len
        updated = True
    if max_size_kb is not None:
        config["max_size_kb"] = max_size_kb
        updated = True

    if updated:
        save_config(config)
    else:
        console.print(
            "[yellow]No changes provided. Use options to set values.[/yellow]"
        )


@app.command()
def view_config():
    """
    This command loads the configuration file and prints its contents to the console.
    If the configuration file is missing or empty, a warning message is displayed.
    """
    config = load_config()
    if not config:
        console.print("[yellow]No configuration file found or it is empty.[/yellow]")
        return
    console.print("[bold blue]Current Configuration:[/bold blue]")
    for key, value in config.items():
        console.print(f"[cyan]{key}:[/cyan] {value}")


if __name__ == "__main__":
    app()
