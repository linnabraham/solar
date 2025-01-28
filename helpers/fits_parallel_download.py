import csv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Function to download a single file, skipping if the file already exists
def download_file(url, output_dir):
    filename = output_dir / url.split("/")[-1]
    if filename.exists():
        return f"File already exists: {filename}"

    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return f"Downloaded: {filename}"
    except Exception as e:
        return f"Failed to download {url}: {e}"

def read_urls_from_csv(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        return [row[0] for row in reader if row]

# Main function to download all URLs in parallel with a progress bar
def download_urls_in_parallel(urls, output_dir, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url, output_dir): url for url in urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading files"):
            result = future.result()  # This will fetch the result or raise exceptions
            print(result)  # Optional: Print the status of each download
