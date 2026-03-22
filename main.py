from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
import shutil
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "model/model0223_v8n.pt")
MODEL_PATH = os.path.join(BASE_DIR, "model/model0303.pt")

model = YOLO(MODEL_PATH)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(file_path)

    if len(results) == 0:
        return {"error": "No detection result"}

    r = results[0]

    names = r.names

    trees = []
    humans = []

    if r.boxes is not None and len(r.boxes) > 0:

        xyxy = r.boxes.xyxy.cpu().tolist()
        cls_ids = r.boxes.cls.cpu().tolist()

        for box, cid in zip(xyxy, cls_ids):

            x1, y1, x2, y2 = box

            label = names[int(cid)].lower()

            height = y2 - y1
            bottom_y = y2
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            obj = {
                "height": float(height),
                "bottom_y": float(bottom_y),
                "center": (center_x, center_y)
            }

            if "tree" in label:
                trees.append(obj)

            elif "human" in label or "person" in label:
                humans.append(obj)


    tree_heights = [t["height"] for t in trees]
    human_heights = [h["height"] for h in humans]

    tallest_human = max(humans, key=lambda x: x["height"]) if humans else None
    highest_tree = max(trees, key=lambda x: x["height"]) if trees else None

    # tallest_human = max(humans, key=lambda x: x["height"]) if humans else None

    # nearest_tree = None

    # if tallest_human and trees:

    #     hx, hy = tallest_human["center"]

    #     min_dist = float("inf")

    #     for t in trees:
    #         tx, ty = t["center"]

    #         dist = math.hypot(tx - hx, ty - hy)

    #         if dist < min_dist:
    #             min_dist = dist
    #             nearest_tree = t

    # output = {
    #     "treeCount": len(trees),
    #     "humanCount": len(humans),
    #     "treeHeights": tree_heights,
    #     "humanHeights": human_heights,
    #     "nearestTreeHeight": nearest_tree["height"] if nearest_tree else None,
    #     "tallestHumanHeight": tallest_human["height"] if tallest_human else None,
    #     "nearestTreeBottomBoundingBoxLineYaxixNumber": nearest_tree["bottom_y"] if nearest_tree else None,
    #     "tallestHumanBottomBoundingBoxLineYaxixNumber": tallest_human["bottom_y"] if tallest_human else None,
    #     "nearestTreeHeight": highest_tree["height"] if highest_tree else None,
    #     "nearestTreeBottomBoundingBoxLineYaxixNumber": highest_tree["bottom_y"] if highest_tree else None,
    # }
    output = {
    "treeCount": len(trees),
    "humanCount": len(humans),
    "treeHeights": tree_heights,
    "humanHeights": human_heights,

    "nearestTreeHeight": highest_tree["height"] if highest_tree else None,
    "nearestTreeBottomBoundingBoxLineYaxixNumber": highest_tree["bottom_y"] if highest_tree else None,

    "tallestHumanHeight": tallest_human["height"] if tallest_human else None,
    "tallestHumanBottomBoundingBoxLineYaxixNumber": tallest_human["bottom_y"] if tallest_human else None,
}

    return output