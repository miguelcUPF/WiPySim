import os, shutil

def get_project_root() -> str:
    """Finds the root directory of the project."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, 'setup.py')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        current_dir = parent_dir

def clean_folder(folder_path):
    """Deletes all files inside the folder if it exists."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Delete folder and all contents
    os.makedirs(folder_path, exist_ok=True)

def get_unique_filename(folder, filename, ext):
    """
    Generates a unique filename considering the existing files in the folder by appending (1), (2), etc. to the filename.

    Parameters
    ----------
    folder (str): Folder path
    filename (str): Desired filename (without extension)
    ext (str): File extension (without dot, e.g., "pdf")

    Returns
    -------
    str: Unique file path.
    """
    file_path = os.path.join(folder, f"{filename}.{ext}")
    if not os.path.exists(file_path):
        return file_path

    # If file exists, append (1), (2), etc.
    counter = 1
    while True:
        new_file_path = os.path.join(folder, f"{filename} ({counter}).{ext}")
        if not os.path.exists(new_file_path):
            return new_file_path
        counter += 1
