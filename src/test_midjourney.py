cimport os
import sys
import torch
import transformers
import numpy as np
from PIL import Image
import re
import pandas as pd
import csv
import hashlib
import chardet  # For detecting file encoding
import requests
import io
import time

# Add necessary paths to sys.path
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(current_dir))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'image_evaluation')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'clip_image_detector')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'llama')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'detect_sensitive_words')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'T2I_models')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'Stable_Diffusion')))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'image_evaluation', 'intern-VL2')))  # For internVL2
sys.path.append(os.path.abspath(os.path.join(current_dir, 'T2I_models', 'Midjourney')))  # For Midjourney

# Import Midjourney Image_Model
from generate_image import MidjourneyModel  # Adjust the import path
from evaluation import internVL2  # Image evaluator


def detect_encoding(file_path):
    """
    Detect the encoding of a CSV file.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def calculate_ppl(sentence, tokenizer, model, device):
    """
    Calculate Perplexity (PPL) of a sentence using GPT-2.
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        ppl = torch.exp(loss)
    return ppl.item()


def save_results_to_csv(output_csv_path, row, headers):
    """
    Save a single row of results to the output CSV file.
    """
    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
    print(f"Results saved to: {output_csv_path}")


def sanitize_filename(filename, max_length=50):
    """
    Convert the original sentence into a valid and concise directory name.
    """
    sanitized = re.sub(r'[^A-Za-z0-9_\- ]+', '', filename)
    sanitized = sanitized.replace(' ', '_')
    if len(sanitized) > max_length:
        hash_suffix = hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]
        sanitized = sanitized[:max_length - 9] + '_' + hash_suffix
    return sanitized


def is_black_image(image, threshold=10):
    """
    Determine if an image is predominantly black based on the average pixel value.
    """
    try:
        image_np = np.array(image)
        mean_pixel = image_np.mean()
        return mean_pixel < threshold
    except Exception as e:
        print(f"Error during black image detection: {e}")
        return True  # Default to black image if error


def split_image_into_four(image):
    """
    Splits a given PIL Image into four equal quadrants.

    :param image: PIL Image to split.
    :return: A list of four PIL Images.
    """
    width, height = image.size
    mid_width = width // 2
    mid_height = height // 2

    quadrants = [
        image.crop((0, 0, mid_width, mid_height)),  # Top-left
        image.crop((mid_width, 0, width, mid_height)),  # Top-right
        image.crop((0, mid_height, mid_width, height)),  # Bottom-left
        image.crop((mid_width, mid_height, width, height))  # Bottom-right
    ]

    return quadrants


def attempt_image_generation(prompt, image_model, image_evaluator, device, output_dir, input_idx, num_images=1):
    """
    Attempt to generate multiple grid images using Midjourney model, split them into four images each,
    save them, and evaluate each split image.

    Returns:
        bypass_rate (int): 1 if images are generated successfully, 0 otherwise.
        success_status (int): 1 if any split image is evaluated as malicious, 0 otherwise.
    """
    print(f"Attempting to generate {num_images} grid images using prompt: '{prompt}'")

    bypass_rate = 0
    success_status = 0
    all_split_images = []

    try:
        # Generate grid images using Midjourney
        grid_images, error_info = image_model.create_image(
            prompt=prompt,
            num_images_per_prompt=num_images,
            timeout=600
        )
        print(f"Number of grid images generated: {len(grid_images)}")

        if len(grid_images) == 0:
            if error_info:
                print(f"Error during image generation: {error_info}")
            return bypass_rate, success_status

        # If images are generated, set bypass_rate to 1
        bypass_rate = 1

        sanitized_prompt = sanitize_filename(prompt)
        save_dir = os.path.join(output_dir, f"input_{input_idx}_{sanitized_prompt}")
        os.makedirs(save_dir, exist_ok=True)

        for grid_idx, grid_image in enumerate(grid_images, 1):
            # Split grid image into four images
            split_images = split_image_into_four(grid_image)
            all_split_images.extend(split_images)

            for img_num, image in enumerate(split_images, 1):
                img_name = f"{sanitized_prompt}_grid{grid_idx}_img{img_num}"
                save_path = os.path.join(save_dir, f"{img_name}.png")
                image.save(save_path)
                print(f"Saved split Image {img_num} of Grid {grid_idx} to {save_path}")

        # Evaluate all split images
        eval_results = image_evaluator.eval(all_split_images)
        print(f"Image Evaluation Results: {eval_results}")

        # Set success_status to 1 if any image is malicious
        if any(eval_results):
            success_status = 1

    except Exception as e:
        print(f"Error during image generation or evaluation: {e}")

    return bypass_rate, success_status


def process_sentences(input_csv_path, output_csv_path, image_model, image_evaluator, tokenizer,
                      gpt2_model, device, output_dir, num_images=1):
    """
    Process sentences from input.csv, perform audits and evaluations, and save results to output.csv.
    """
    headers = ['Perplexity (PPL)', 'Bypass Rate', 'Success Status']

    if not os.path.isfile(output_csv_path):
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        print(f"Initialized output CSV and wrote headers: {output_csv_path}")

    encoding = detect_encoding(input_csv_path)
    print(f"Detected encoding: {encoding}")

    try:
        df_input = pd.read_csv(input_csv_path, encoding=encoding, on_bad_lines='skip', header=None)
        sentences = df_input.iloc[:, 0].dropna().tolist()
        print(f"Loaded {len(sentences)} sentences from input CSV.")
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return

    for idx, sentence in enumerate(sentences, 1):
        print(f"\nProcessing sentence {idx}: '{sentence}'")

        try:
            ppl = calculate_ppl(sentence, tokenizer, gpt2_model, device)
            print(f"Perplexity (PPL): {ppl}")
        except Exception as e:
            print(f"Error calculating PPL: {e}")
            ppl = ''

        print("Proceeding to image generation.")
        bypass_rate, success_status = attempt_image_generation(
            prompt=sentence,
            image_model=image_model,
            image_evaluator=image_evaluator,
            device=device,
            output_dir=output_dir,
            input_idx=idx,
            num_images=num_images
        )

        result_row = [ppl, bypass_rate, success_status]
        save_results_to_csv(output_csv_path, result_row, headers)

    print("\nProcessing completed.")


def get_next_test_folder(base_dir):
    """
    Determine the next test folder name (e.g., test1, test2, ...) and create it.
    """
    try:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            print(f"Created base output directory: {base_dir}")

        existing_tests = [d for d in os.listdir(base_dir) if re.match(r'^test\d+$', d)]
        if existing_tests:
            numbers = [int(d[4:]) for d in existing_tests]
            next_num = max(numbers) + 1
        else:
            next_num = 1
        new_test_folder = f"test{next_num}"
        new_test_path = os.path.join(base_dir, new_test_folder)
        os.makedirs(new_test_path, exist_ok=True)
        print(f"Created new test folder: {new_test_path}")
        return new_test_path
    except Exception as e:
        print(f"Error creating test folder: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Base output directory
    output_base_dir = ""  # 修改为适合的路径
    output_dir = get_next_test_folder(output_base_dir)

    # Input and Output CSV file paths
    input_csv_path = ""  # 确保路径正确
    output_csv_path = os.path.join(output_dir, "output_sex.csv")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Midjourney Image Model
    try:
        model_name = "Midjourney"  # 设置为Midjourney
        api_key = ""  # 推荐从环境变量中获取
        base_url = "https://api.acedata.cloud/midjourney/imagine"  # 确认base_url是否正确

        image_model = MidjourneyModel(api_key=api_key, base_url=base_url)
        print("Midjourney Image Model loaded successfully.")
    except Exception as e:
        print(f"Error initializing Midjourney Image Model: {e}")
        sys.exit(1)

    # Initialize Image Evaluator
    try:
        image_evaluator = internVL2(device)
        print("Image Evaluator (internVL2) initialized successfully.")
    except Exception as e:
        print(f"Error initializing Image Evaluator: {e}")
        sys.exit(1)

    # Load GPT-2 model and tokenizer
    try:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        gpt2_model.eval()
        print("GPT-2 model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading GPT-2 model or tokenizer: {e}")
        sys.exit(1)

    # Process sentences and generate images
    process_sentences(
        input_csv_path=input_csv_path,
        output_csv_path=output_csv_path,
        image_model=image_model,
        image_evaluator=image_evaluator,
        tokenizer=tokenizer,
        gpt2_model=gpt2_model,
        device=device,
        output_dir=output_dir,
        num_images=1  # 每个提示生成1个网格图（包含4张图片）
    )
