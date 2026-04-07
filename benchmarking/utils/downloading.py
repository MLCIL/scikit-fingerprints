"""
Download utils (same as in the Molpile repo)
"""

import os
import shutil
import subprocess
import zipfile

from rapidgzip import RapidgzipFile


def download_single_file(
    url: str,
    output_dir: str,
    output_file: str,
) -> str:
    output_path = os.path.join(output_dir, output_file)
    cmd = [
        "wget",
        "-c",
        "-O",
        os.path.abspath(output_path),
        url,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def download_multiple_files(
    urls: list[str],
    output_dir: str,
) -> str:
    for url in urls:
        filename = url.split("/")[-1]
        output_path = os.path.join(output_dir, filename)
        cmd = [
            "wget",
            "-c",
            "-O",
            os.path.abspath(output_path),
            url,
        ]
        subprocess.run(cmd, check=True)

    return output_dir


def unpack_archive(file_path: str) -> None:
    extension = file_path.split(".")[-1]

    if extension == "gz":
        output_path = file_path[:-3]
        with RapidgzipFile(file_path) as f_in, open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    elif extension == "zip":
        directory = os.path.dirname(file_path)
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(directory)
    else:
        raise ValueError(f"Archive extension {extension} not supported")
