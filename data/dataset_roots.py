import os


def get_sysu_root():
    search_root = "/kaggle/input"

    candidates = []

    for root, dirs, files in os.walk(search_root):
        if os.path.basename(root) == "SYSU-MM01":
            candidates.append(root)

    if len(candidates) == 0:
        raise FileNotFoundError("Could not find SYSU-MM01 inside /kaggle/input")

    sysu_root = candidates[0]

    required = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "exp"]

    for item in required:
        path = os.path.join(sysu_root, item)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required folder: {path}")

    return sysu_root
