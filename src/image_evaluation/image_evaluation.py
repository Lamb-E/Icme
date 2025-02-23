from clip_image_detector.predict import clip_predict_image
from Q16.evaluation import q16_predict_image
from PIL import Image
import torch


def load_images(img_path):
    images = []
    if type(img_path) == list:
        for path in img_path:
            image = Image.open(path)
            images.append(image)
    else:
        image = Image.open(img_path)
        images.append(image)
    return images


if __name__ == "__main__":
    image_path = ""
    images = load_images(image_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_eva = clip_predict_image(device)
    q16_eva = q16_predict_image(device)
    res1 = clip_eva.get_image_result(images=images)
    res2 = q16_eva.get_image_result(images=images)
    print(f"clip eva: {res1}, q16 eva: {res2}")
