import os
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
sys.path.append(os.path.abspath(os.path.join(current_dir, 'T2I_models', 'DALLE3')))  # For DALLE3

t2i_models_path = os.path.abspath(os.path.join(current_dir, 'T2I_models'))
if t2i_models_path not in sys.path:
    sys.path.append(t2i_models_path)

# Import DALLE3 Image_Model
from dalle3_single_generation import Image_Model  # Adjust the import path
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


def attempt_image_generation(prompt, image_model, image_evaluator, device, output_dir, input_idx, num_images=4):
    """
    尝试使用DALLE3模型逐一生成多张图片，并使用internVL2进行评估。
    返回:
        bypass_rate (int): 成功生成的图片数量。
        success_status (int): 如果任何一张图片被评估为恶意，返回1，否则返回0。
    """
    print(f"Attempting to generate {num_images} images using prompt: '{prompt}'")

    bypass_rate = 0
    success_status = 0

    sanitized_prompt = sanitize_filename(prompt)
    save_dir = os.path.join(output_dir, f"input_{input_idx}_{sanitized_prompt}")
    os.makedirs(save_dir, exist_ok=True)

    for img_idx in range(1, num_images + 1):
        try:
            # 逐一生成单张图片
            images, error_info, revised_prompts = image_model.create_image(
                prompt,
                num_images_per_prompt=1,
            )

            if len(images) == 0:
                print(f"Error during image generation for image{img_idx}: {error_info[0].error['message']}")
                continue  # 跳过当前图片，继续生成下一张

            image = images[0]
            bypass_rate += 1

            # 处理修订后的提示（如果有）
            if revised_prompts and img_idx <= len(revised_prompts):
                img_name = revised_prompts[0].encode('utf-16', 'surrogatepass').decode('utf-16')[:80]
            else:
                img_name = sanitized_prompt

            save_path = os.path.join(save_dir, f"image{img_idx}.png")
            image.save(save_path)
            print(f"Saved Image {img_idx} to {save_path}")

            # 评估生成的图片
            eval_results = image_evaluator.eval([image])
            print(f"Image {img_idx} Evaluation Results: {eval_results}")

            # 如果任何一张图片被评估为恶意，设置success_status为1
            if any(eval_results):
                success_status = 1

        except Exception as e:
            print(f"Error during image generation or evaluation for image{img_idx}: {e}")
            continue  # 继续生成下一张图片

    return bypass_rate, success_status


def process_sentences(input_csv_path, output_csv_path, image_model, image_evaluator, tokenizer,
                      gpt2_model, device, output_dir, num_images=4):
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
    output_base_dir = ""
    output_dir = get_next_test_folder(output_base_dir)

    # Input and Output CSV file paths
    input_csv_path = ""
    output_csv_path = os.path.join(output_dir, "output_blood.csv")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize DALLE3 Image Model
    try:
        model_name = "DALLE3"  # Options: ["DALLE3", "DALLE2", "SDXL", "SD3"]
        api_key = ""  # Replace with your actual API key
        base_url = ""  # Replace with your actual base URL if different

        image_model = Image_Model(model_name=model_name, api_key=api_key, base_url=base_url)
        print("DALLE3 Image Model loaded successfully.")
    except Exception as e:
        print(f"Error initializing DALLE3 Image Model: {e}")
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
        num_images=4  # Set to generate four images per input
    )
