from pathlib import Path

"""
To run: navigate to the folder this file is in in the terminal
type python3 generate_tree.py and hit enter. The script should run
automatically and output a file containing the project tree structure
"""

EXCLUDE = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".DS_Store",
    "venv",
    ".venv",
    "wdbc-env",
    "tmp",
    "uv.lock",
    # ".github",
    # ".gitignore",
    ".gitkeep"
}

def generate_tree(
        directory: Path, 
        prefix: str = "",
        depth: int = 0, # current recursion depth
        max_depth: int | None = None # None => unlimited
        ) -> list[str]:
    """
    Return a list of strings representing the directory tree.

    Parameters: 
    -----------
    directory : Path
        Folder whose contents we are listing
    prefix : str
        Visual prefix used for the tree branches
    depth : int
        Current recursion depth (used internally)
    max_depth : int | None
        Maximum depth to descend into. `None` means "no limit".
    """
    # Stop recursing once we hit the limit
    if max_depth is not None and depth >= max_depth:
        return []
    
    # Gather children, ignoring anthing in EXCLUDE
    contents = sorted(
        [item for item in directory.iterdir() if item.name not in EXCLUDE],
        key=lambda x: (x.is_file(), x.name.lower())
    )

    # Tree branch symbols
    pointers = ["├── "] * (len(contents) - 1) + ["└── "]

    lines: list[str] = []

    for pointer, path in zip(pointers, contents):
        line = prefix + pointer + path.name
        lines.append(line)

        # If it's a directory, recurse one level deeper
        if path.is_dir():
            extension = "│   " if pointer == "├── " else "    "
            lines.extend(
                generate_tree(
                    path, 
                    prefix + extension,
                    depth = depth + 1,
                    max_depth = max_depth
                )
            )

    return lines


if __name__ == "__main__":
    # Example: limit the tree to 2 levels deep (root + one sub-folder level)
    root = Path("../../")
    tree_lines = [root.resolve().name + "/"]
    tree_lines.extend(generate_tree(root, max_depth=2))

    with open("PROJECT_STRUCTURE.md", "w") as f:
        f.write("```\n")
        f.write("\n".join(tree_lines))
        f.write("\n```")

    print("Project structure written to PROJECT_STRUCTURE.md")