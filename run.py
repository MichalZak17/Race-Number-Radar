
import io
import re
import os
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

SITE_URL            = "http://localhost"
SITE_NAME           = "RaceNumberRadar"
API_MODEL           = "qwen/qwen2.5-vl-32b-instruct"
WORKERS             = 64
BIB_NUMBER_LENGTHS  = 3
PROVIDER            = "deepinfra/bf16"

app                 = typer.Typer()
console             = Console()

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

def scan_directory(directory: str) -> list[str]:
    """
    Scans the specified directory and returns a list of image file paths with supported extensions.

    Args:
        directory (str): The path to the directory to scan.

    Returns:
        list[str]: A list of file paths for images with extensions .jpg, .jpeg, .png, or .webp.
    """
    files = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for item in os.scandir(directory):
        if item.is_file() and os.path.splitext(item.name)[1].lower() in supported_extensions:
            files.append(item.path)
    return files

def image_to_base64_uri(image: Image.Image, max_size_kb=1500) -> str:
    """
    Converts a PIL Image to a base64-encoded JPEG URI, optionally compressing to fit within a specified size.
    Args:
        image (Image.Image): The PIL Image to convert.
        max_size_kb (int, optional): Maximum allowed size of the encoded image in kilobytes. Defaults to 1500.
    Returns:
        str: A data URI containing the base64-encoded JPEG image.
    Notes:
        The function reduces JPEG quality in steps of 5 until the encoded image fits within max_size_kb or quality reaches 10.
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

def process_single_image(file_path: str, client: OpenAI) -> tuple[str, list[str]]:
    """
    Processes a single image to identify race bib numbers using an OpenAI client.
    Args:
        file_path (str): The path to the image file to be processed.
        client (OpenAI): An instance of the OpenAI client for making API requests.
    Returns:
        tuple[str, list[str]]: A tuple containing the file path and a list of detected race bib numbers as strings.
            If no numbers are found or an error occurs, the list will be empty.
    Raises:
        APIError: If an error occurs during the API request.
        Exception: For any other errors encountered during processing.
    Notes:
        - The function sends the image and a prompt to the OpenAI API to extract race bib numbers.
        - If no numbers are found, or if the response contains 'none', an empty list is returned.
        - Errors are printed to the console and an empty list is returned in case of failure.
    """
    try:
        with Image.open(file_path).convert("RGB") as image:
            base64_image_uri = image_to_base64_uri(image)

        prompt_text = (
            "Identify all race bib numbers in this image. "
            "Respond with only the numbers, separated by commas. "
            "For example: 123,431,890. The bib numbers are alwasy 3 digits. If no numbers are found, respond with 'none'."
        )

        extra_body_params = {}
        extra_body_params["provider"] = {"only": [PROVIDER]}

        completion = client.chat.completions.create(
            
            extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_NAME},
            model=API_MODEL,
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
            extra_body=extra_body_params
        )
        
        response_text = completion.choices[0].message.content.strip().lower()
        
        if "none" in response_text or not response_text:
            return file_path, []

        return file_path, re.findall(r'\d+', response_text)

    except APIError as e:
        console.print(f"[bold red]API Error for {os.path.basename(file_path)}:[/bold red] {e.message}")
    except Exception as e:
        console.print(f"[bold red]Error processing {os.path.basename(file_path)}:[/bold red] {e}")
    return file_path, []


@app.command()
def process(directory: str = typer.Argument(..., help="Directory with images to process.")):
    """
    Processes a directory of images to detect race numbers using an OpenAI-powered client, organizes images by detected numbers, and copies them into corresponding subdirectories.
    Args:
        directory (str): Path to the directory containing images to process.
    Workflow:
        1. Scans the specified directory for image files.
        2. Initializes the OpenAI client for number detection.
        3. Processes each image concurrently, extracting detected numbers.
        4. Filters detected numbers to those with 3 or 4 digits.
        5. Organizes images by detected number, copying them into subdirectories named after each number.
    Outputs:
        - Progress and status messages to the console.
        - Images organized into subdirectories by detected race number.
    Raises:
        - Prints error messages to the console if API client initialization fails or if image copying encounters issues.
    """
    files = scan_directory(directory)
    if not files:
        console.print(f"[bold red]No image files found in {directory}.[/bold red]")
        return
        
    console.print(f"Processing [bold blue]{len(files)}[/bold blue] files in [bold yellow]{directory}[/bold yellow] with {WORKERS} workers...")

    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize API client: {e}[/bold red]")
        return

    number_to_images = defaultdict(list)
    
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(files))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_to_file = {executor.submit(process_single_image, file_path, client): file_path for file_path in files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path, detected_numbers = future.result()
                
                if detected_numbers:
                    found_valid_number = False
                    for number in detected_numbers:
                        if 3 <= len(number) <= 4:
                            number_to_images[number].append(file_path)
                            found_valid_number = True
                    if found_valid_number:
                        progress.print(f"File: [green]{os.path.basename(file_path)}[/green] -> Detected: [bold cyan]{', '.join(detected_numbers)}[/bold cyan]")
                
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

if __name__ == "__main__":
    app()
