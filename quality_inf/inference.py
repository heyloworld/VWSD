from model import  *
from prompts import get_prompt, get_cot_prompt, get_tot_prompt
import argparse
import json
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='Run different types of inference')
    parser.add_argument('--inference_type', type=str, required=True, 
                      choices=['general', 'cot', 'tot'],
                      help='Type of inference to run (general, cot, or tot)')
    parser.add_argument('--test_path', type=str, 
                      default='./datasets/VWSD/queries/en.test.data.v1.1.txt',
                      help='Path to test data')
    parser.add_argument('--gold_path', type=str,
                      default='./queries/en.test.gold.v1.1.txt',
                      help='Path to gold data')
    parser.add_argument('--img_path', type=str,
                      default='./test_images_resized',
                      help='Path to image directory')
    parser.add_argument('--max_samples', type=int, default=100,
                      help='Maximum number of samples to process')
    return parser.parse_args()

def get_inference(test_data, gold_data, args):
    model, tokenizer = load_model()
    all_results = []

    for i in range(len(test_data)):
        if i > args.max_samples:
            break
            
        context = test_data[i][0]
        query = test_data[i][1]
        candidate_images = test_data[i][2:]
        gold_img = gold_data[i]
        img_list = [f"{args.img_path}/{img}" for img in candidate_images]
        
        entry = {
            "context": context,
            "query": query,
            "responses": []
        }

        generation_config = dict(max_new_tokens=1096, do_sample=True)
        PROMPT = f"Can you caption this image with the '{query}' given the '{context}'or not? Return Yes or No"

        for img in img_list:
            pixel_values = load_image(img, max_num=12).to(torch.bfloat16).cuda()
            response, history = model.chat(tokenizer, pixel_values, PROMPT, generation_config, history=None, return_history=True)
            entry["responses"].append({
                "image": img,
                "gold_img": gold_img,
                "response": response
            })
        
        all_results.append(entry)

    with open('general_inference_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

def get_cot_inference(test_data, gold_data, args):
    model, tokenizer = load_model()
    all_results = []

    for i in range(len(test_data)):
        if i > args.max_samples:
            break
            
        context = test_data[i][0]
        query = test_data[i][1]
        candidate_images = test_data[i][2:]
        gold_img = gold_data[i]
        img_list = [f"{args.img_path}/{img}" for img in candidate_images]
        
        entry = {
            "context": context,
            "query": query,
            "gold_img": gold_img,
            "responses": []
        }

        generation_config = dict(max_new_tokens=1096, do_sample=True)
        PROMPT = get_cot_prompt(query, context)

        for img in img_list:
            pixel_values = load_image(img, max_num=12).to(torch.bfloat16).cuda()
            response, history = model.chat(tokenizer, pixel_values, PROMPT, generation_config, history=None, return_history=True)
            entry["responses"].append({
                "image": img,
                "response": response
            })
        
        all_results.append(entry)

    with open('cot_inference_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

def get_tot_inference(test_data, gold_data, args):
    model, tokenizer = load_model()
    all_results = []

    for i in range(len(test_data)):
        if i > args.max_samples:
            break
            
        context = test_data[i][0]
        query = test_data[i][1]
        candidate_images = test_data[i][2:]
        gold_img = gold_data[i]
        img_list = [f"{args.img_path}/{img}" for img in candidate_images]
        
        entry = {
            "context": context,
            "query": query,
            "gold_img": gold_img,
            "responses": []
        }

        generation_config = dict(max_new_tokens=1096, do_sample=True)
        PROMPT = get_tot_prompt(query, context)

        for img in img_list:
            pixel_values = load_image(img, max_num=12).to(torch.bfloat16).cuda()
            response, history = model.chat(tokenizer, pixel_values, PROMPT, generation_config, history=None, return_history=True)
            entry["responses"].append({
                "image": img,
                "response": response
            })
        
        all_results.append(entry)

    with open('tot_inference_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

def main():
    args = parse_args()

    with open(args.test_path, "r", encoding="utf-8") as file:
        test_data = [line.strip().split("\t") for line in file]

    with open(args.gold_path, "r", encoding="utf-8") as file:
        gold_data = [line.strip().split("\t") for line in file]

    if args.inference_type == 'general':
        get_inference(test_data, gold_data, args)
    elif args.inference_type == 'cot':
        get_cot_inference(test_data, gold_data, args)
    elif args.inference_type == 'tot':
        get_tot_inference(test_data, gold_data, args)

if __name__=='__main__':
    main()