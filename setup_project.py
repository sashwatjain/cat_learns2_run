import os

# Root directory (where this script is placed)
ROOT = os.path.dirname(os.path.abspath(__file__))

# Folder structure definition
structure = {
    "env": [
        "cat_env.py",
        "cat_model.py"
    ],
    "train": [
        "train_ppo.py"
    ],
    "utils": [
        "logger.py"
    ],
    "assets": [
        "placeholder.txt"
    ]
}

def create_structure():
    print("Creating project folders...\n")

    for folder, files in structure.items():
        folder_path = os.path.join(ROOT, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")

        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("")   # empty file
            print(f"  Created file: {file_path}")

    print("\nDone! Folder structure created successfully.")

if __name__ == "__main__":
    create_structure()
