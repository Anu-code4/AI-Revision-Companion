

import hashlib
import json
from pathlib import Path


# ==========================================================
# Paths
# ==========================================================

DOCUMENTS_PATH = Path("Document")
METADATA_FILE = Path("metadata/documents.json")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


# ==========================================================
# Compute SHA-256 Hash
# ==========================================================

def compute_file_hash(file_path: Path) -> str:
    """
    Computes the SHA-256 hash of a file.

    Args:
        file_path: Path of the document.

    Returns:
        SHA-256 hash string.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while chunk := file.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


# ==========================================================
# Metadata
# ==========================================================

def load_metadata() -> dict:
    """
    Load document metadata.
    """

    if not METADATA_FILE.exists():
        return {}

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_metadata(metadata: dict):
    """
    Save document metadata.
    """

    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


# ==========================================================
# Scan Documents
# ==========================================================

def scan_documents() -> list[Path]:
    """
    Scan supported documents from the Document folder.
    """

    documents = []

    for file in DOCUMENTS_PATH.iterdir():

        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(file)

    return documents


# ==========================================================
# Detect Changes
# ==========================================================

def detect_changes():
    """
    Detect new, modified and unchanged documents.
    """

    metadata = load_metadata()

    new_files = []
    modified_files = []
    unchanged_files = []

    for file in scan_documents():

        current_hash = compute_file_hash(file)

        old_info = metadata.get(file.name)

        if old_info is None:

            new_files.append(file)

        elif old_info["hash"] != current_hash:

            modified_files.append(file)

        else:

            unchanged_files.append(file)

    return (
        new_files,
        modified_files,
        unchanged_files,
    )


# ==========================================================
# Update Metadata
# ==========================================================

def update_metadata():
    """
    Update metadata after successful indexing.
    """

    metadata = {}

    for file in scan_documents():

        metadata[file.name] = {
            "hash": compute_file_hash(file),
            "path": str(file),
        }

    save_metadata(metadata)


# ==========================================================
# Deleted Files
# ==========================================================

def detect_deleted_files():
    """
    Return deleted files.
    """

    metadata = load_metadata()

    current_files = {
        file.name
        for file in scan_documents()
    }

    deleted_files = []

    for filename in metadata:

        if filename not in current_files:
            deleted_files.append(filename)

    return deleted_files


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    new_files, modified_files, unchanged_files = detect_changes()

    print("\nNew Files")
    for file in new_files:
        print("-", file.name)

    print("\nModified Files")
    for file in modified_files:
        print("-", file.name)

    print("\nUnchanged Files")
    for file in unchanged_files:
        print("-", file.name)

    deleted_files = detect_deleted_files()

    print("\nDeleted Files")
    for file in deleted_files:
        print("-", file)

    update_metadata()

    print("\nMetadata updated successfully.")