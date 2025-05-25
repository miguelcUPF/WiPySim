import os, shutil


def get_project_root() -> str:
    """
    Finds the root directory of the project.

    Returns:
        str: The absolute path of the project root.

    Raises:
        FileNotFoundError: If 'requirements.txt' is not found within a reasonable number of parent directories.
    """

    current_directory = os.getcwd()
    max_iterations = 1000

    for _ in range(max_iterations):
        if os.path.exists(os.path.join(current_directory, "requirements.txt")):
            return current_directory

        parent_directory = os.path.dirname(current_directory)

        if parent_directory == current_directory:
            break

        current_directory = parent_directory

    raise FileNotFoundError(
        f"Could not find 'requirements.txt' or reached root directory"
    )


def clean_folder(folder_path: str):
    """
    Deletes all files inside the folder if it exists.

    Args:
        folder_path (str): The path to the folder.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Delete folder and all contents
    os.makedirs(folder_path, exist_ok=True)


def get_unique_filename(folder_path: str, filename: str, extension: str) -> str:
    """
    Generates a unique filename considering the existing files in the folder by appending (1), (2), etc. to the filename.

    Args:
        folder_path (str): Folder path
        filename (str): Desired filename (without extension)
        extension (str): File extension (without dot, e.g., "pdf")

    Returns:
        str: Unique file path.
    """
    file_path = os.path.join(folder_path, f"{filename}.{extension}")
    if not os.path.exists(file_path):
        return file_path

    # If file exists, append (1), (2), etc.
    counter = 1
    while True:
        new_file_path = os.path.join(folder_path, f"{filename} ({counter}).{extension}")
        if not os.path.exists(new_file_path):
            return new_file_path
        counter += 1
