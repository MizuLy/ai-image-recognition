"""
Task 1 - Web version
AI Image Analysis with upload form (Flask + CLIP)

CLIP is "open-vocabulary" - unlike YOLO, it isn't limited to 80 fixed
classes. It compares the image against ANY list of text labels you
give it and returns how well each one matches. That's why it can
recognize things like "dinosaur" or "anime character" that YOLO can't.

Supports both file upload AND pasting an image URL directly.

Run with:
    python app.py
Then open http://127.0.0.1:8000 in your browser.
"""

import os
import requests
import clip
import torch
from PIL import Image
from io import BytesIO
from flask import Flask, request, render_template, url_for

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model once at startup (faster than reloading every request)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Candidate labels CLIP will choose between. Edit this list to add
# more categories relevant to your test images.
LABELS = [
    # People
    "a person", "a group of people", "a child", "a baby", "a face close-up",
    "a crowd", "a portrait", "a person's hands",

    # Common animals
    "a Golden Retriever", "a Labrador", "a German Shepherd", "a Poodle",
    "a Bulldog", "a Husky", "a Chihuahua", "a Corgi", "a Beagle", "a dog",
    "a cat", "a Siamese cat", "a Persian cat", "a horse", "a cow",
    "a sheep", "a pig", "a rabbit", "a fish", "a goldfish",
    "a chicken", "a duck", "a mouse", "a hamster", "a guinea pig",

    # Wild / exotic animals
    "a lion", "a tiger", "an elephant", "a bear", "a wolf", "a fox",
    "a monkey", "a gorilla", "a chimpanzee", "a giraffe", "a zebra",
    "a rhino", "a hippo", "a cheetah", "a leopard", "a jaguar",
    "a kangaroo", "a koala", "a panda", "a sloth", "a raccoon",
    "a deer", "a moose", "a bison", "a camel", "a bat",
    "a snake", "a lizard", "a gecko", "an iguana", "a chameleon",
    "a crocodile", "an alligator", "a frog", "a toad", "a turtle",
    "a tortoise", "a shark", "a whale", "a dolphin", "a seal",
    "a bee", "an ant", "an insect", "a butterfly", "a beetle",
    "a scorpion", "a spider", "a crab", "a lobster", "an octopus",
    "a jellyfish", "a starfish", "a snail", "a worm",

    # Dinosaurs (specific species)
    "a Tyrannosaurus rex", "a Triceratops", "a Velociraptor",
    "a Stegosaurus", "a Brachiosaurus", "a Pterodactyl",
    "a Spinosaurus", "an Ankylosaurus", "a Diplodocus", "a Raptor",
    "a dinosaur",

    # Fictional / stylized
    "an anime character", "a cartoon character",
    "a fictional creature", "a monster", "a superhero", "a robot",
    "a video game character", "a fantasy creature", "a dragon",
    "a zombie", "an alien", "a ghost",

    # Vehicles
    "a sedan car", "an SUV", "a sports car", "a pickup truck",
    "a van", "a taxi", "a police car", "a race car", "a truck",
    "a bus", "a motorcycle", "a scooter", "a bicycle",
    "an airplane", "a fighter jet", "a helicopter", "a train",
    "a subway train", "a boat", "a sailboat", "a ship", "a cruise ship",
    "a rocket", "a tractor",

    # Birds (specific species)
    "an eagle", "an owl", "a parrot", "a peacock", "a flamingo",
    "a swan", "a hawk", "a falcon", "a crow", "a raven", "a sparrow",
    "a pigeon", "a robin", "a hummingbird", "a woodpecker",
    "a toucan", "a pelican", "a seagull", "a heron", "a stork",
    "a vulture", "a chicken", "a duck", "an ostrich", "a penguin",
    "a bird",

    # Buildings / places
    "a building", "a house", "a skyscraper", "a bridge", "a castle",
    "a temple", "a church", "a stadium", "a factory", "a street",
    "a city skyline", "a village",

    # Nature / landscape
    "a tree", "a forest", "a rose", "a sunflower", "a tulip",
    "a lily", "an orchid", "a daisy", "a lotus flower", "a cherry blossom",
    "a flower", "a garden", "a mountain",
    "the ocean", "a beach", "a river", "a lake", "a waterfall",
    "a desert", "a sky with clouds", "a sunset", "snow", "a field",
    "a landscape",

    # Fruits
    "an apple", "a banana", "an orange", "a mango", "a watermelon",
    "a strawberry", "grapes", "a pineapple", "a pear", "a peach",
    "a lemon", "a lime", "a kiwi fruit", "a papaya", "a coconut",
    "a pomegranate", "a fruit",

    # Vegetables
    "broccoli", "spinach", "a carrot", "a potato", "a tomato",
    "an onion", "garlic", "a cucumber", "a bell pepper", "a chili pepper",
    "corn", "a mushroom", "lettuce", "cabbage", "a pumpkin",
    "an eggplant", "a vegetable",

    # Prepared food
    "a salad",
    "a cake", "a pizza", "a burger", "sushi", "pasta", "a sandwich",
    "a taco", "fried rice", "soup", "a drink", "a dish of noodles",
    "ice cream", "a dessert", "a coffee cup", "bread", "food",

    # Everyday objects
    "a laptop", "a phone", "a computer", "a television", "a camera",
    "a book", "a chair", "a table", "furniture", "a clock",
    "a bag", "shoes", "clothing", "a toy", "a musical instrument",
    "a bottle", "a cup", "glasses",

    # Sports / activities
    "a sports ball", "a person playing sports", "a person exercising",
    "a person dancing", "a person cooking", "a person swimming",

    # Art / media
    "a painting", "a drawing", "a sculpture", "a photograph",
    "a comic panel", "a poster", "text or a sign", "a chart or graph",
    "a screenshot",

    # Weather / misc
    "rain", "a storm", "fire", "smoke", "an explosion",
]


def classify_image(pil_image):
    image = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
    text = clip.tokenize(LABELS).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    ranked = sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)
    return ranked[:5]  # top 5 matches


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.jpg")
    image_url = request.form.get("image_url", "").strip()
    file = request.files.get("image")

    try:
        if image_url:
            # Fetch image from the pasted URL
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            pil_image = Image.open(BytesIO(resp.content))
            pil_image.convert("RGB").save(input_path)
        elif file and file.filename != "":
            file.save(input_path)
            pil_image = Image.open(input_path)
        else:
            return render_template("index.html", results=None,
                                   error="Please choose a file or paste an image URL.")
    except Exception as e:
        return render_template("index.html", results=None,
                               error=f"Couldn't load that image: {e}")

    top_matches = classify_image(pil_image)

    results_data = {
        "top_label": top_matches[0][0],
        "top_confidence": top_matches[0][1] * 100,
        "matches": [(label, prob * 100) for label, prob in top_matches],
        "input_image": url_for("static", filename="uploads/input.jpg"),
    }

    return render_template("index.html", results=results_data, error=None)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
