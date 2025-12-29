import gdown
import os

from pathlib import Path
from zipfile import ZipFile

#===============================
if __name__ == "__main__":
    url = "https://drive.google.com/file/d/1gHSM4vxC_DLJzNFtxv_-CC4zBHS_hqhT/view?usp=sharing"
    zippath = Path("./pretrained.zip")

    try:
        gdown.download(url=url, output=str(zippath), fuzzy=True, use_cookies=True)
        print(f"Successfully downloaded folder to: {zippath}")

        extractpath = Path("./")
        extractpath.mkdir(parents=True, exist_ok=True)
        with ZipFile(zippath, "r") as zip_file:
            zip_file.extractall(path=extractpath)
        os.remove(zippath)

    except Exception as e:
        print(f"An error occurred: {e}")
        