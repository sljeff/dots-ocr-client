import os
import json
from multiprocessing.pool import ThreadPool, Pool
import argparse


from dots_ocr_client.model.inference import inference_with_vllm
from dots_ocr_client.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr_client.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr_client.utils.doc_utils import fitz_doc_to_image, load_images_from_pdf
from dots_ocr_client.utils.prompts import dict_promptmode_to_prompt
from dots_ocr_client.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr_client.utils.format_transformer import layoutjson2md


class DotsOCRParser:
    """
    parse image or pdf file
    """
    
    def __init__(self, 
            backend="vllm",
            base_url="http://127.0.0.1:8000",
            api_token=None,
            model_name='model',
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=16384,
            num_thread=64,
            dpi = 200, 
            min_pixels=None,
            max_pixels=None,
            replicate_deployment=None,
        ):
        self.dpi = dpi

        # backend configuration
        self.backend = backend
        self.base_url = base_url
        self.api_token = api_token
        self.model_name = model_name
        self.replicate_deployment = replicate_deployment
        
        # default args for inference
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if self.backend == "replicate":
            used = f"deployment '{self.replicate_deployment}'" if self.replicate_deployment else "public model sljeff/dots.ocr"
            print(f"use replicate backend ({used}), num_thread={self.num_thread}")
        else:
            print(f"use vllm model, num_thread will be set to {self.num_thread}")
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS



    def _inference_with_vllm(self, image, prompt):
        response = inference_with_vllm(
            image,
            prompt, 
            base_url=self.base_url,
            api_token=self.api_token,
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def _inference_with_replicate(self, image, prompt):
        from dots_ocr_client.model.inference import inference_with_replicate
        return inference_with_replicate(
            image,
            prompt,
            deployment=self.replicate_deployment,
            api_token=self.api_token,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )

    def _inference(self, image, prompt):
        if self.backend == "replicate":
            return self._inference_with_replicate(image, prompt)
        return self._inference_with_vllm(image, prompt)

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt

    # def post_process_results(self, response, prompt_mode, save_dir, save_name, origin_image, image, min_pixels, max_pixels)
    def _parse_single_image(
        self, 
        origin_image, 
        prompt_mode, 
        save_dir, 
        save_name, 
        source="image", 
        page_idx=0, 
        bbox=None,
        fitz_preprocess=False,
        ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS  # preprocess image to the final input
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: assert max_pixels <= MAX_PIXELS, f"max_pixels should <+ {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)
        response = self._inference(image, prompt)
        result = {'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response, 
                prompt_mode, 
                origin_image, 
                image,
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
                )
            if filtered and prompt_mode != 'prompt_layout_only_en':  # model output json failed, use filtered process
                result.update({
                    'cells': response,
                    'filtered': True
                })
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                result.update({
                    'cells': cells,
                    'image_with_layout': image_with_layout,
                })
                if prompt_mode != "prompt_layout_only_en":  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True) # used for clean output or metric of omnidocbench、olmbench 
                    result.update({
                        'md_content': md_content,
                        'md_content_no_hf': md_content_no_hf,
                    })
        else:
            result.update({
                'response': response,
                'origin_image': origin_image
            })

        return result
    
    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess)
        result['file_path'] = input_path
        return [result]
        
    def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        print(f"loading pdf: {input_path}")
        images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
        total_pages = len(images_origin)
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source":"pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        num_thread = min(total_pages, self.num_thread)
        print(f"Parsing PDF with {total_pages} pages using {num_thread} threads...")

        results = []
        with ThreadPool(num_thread) as pool:
            for result in pool.imap_unordered(_execute_task, tasks):
                results.append(result)
                print(f"Processed page {result['page_no']+1}/{total_pages}")

        results.sort(key=lambda x: x["page_no"])
        for i in range(len(results)):
            results[i]['file_path'] = input_path
        return results

    def parse_file(self, 
        input_path, 
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False
        ):
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = None  # Not used anymore since we don't save files

        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox, fitz_preprocess=fitz_preprocess)
        else:
            raise ValueError(f"file extension {file_ext} not supported, supported extensions are {image_extensions} and pdf")
        
        print(f"Parsing finished")
        return results



def main():
    prompts = list(dict_promptmode_to_prompt.keys())
    parser = argparse.ArgumentParser(
        description="dots.ocr Multilingual Document Layout Parser",
    )
    
    parser.add_argument(
        "input_path", type=str,
        help="Input PDF/image file path"
    )
    

    
    parser.add_argument(
        "--prompt", choices=prompts, type=str, default="prompt_layout_all_en",
        help="prompt to query the model, different prompts for different tasks"
    )
    parser.add_argument(
        '--bbox', 
        type=int, 
        nargs=4, 
        metavar=('x1', 'y1', 'x2', 'y2'),
        help='should give this argument if you want to prompt_grounding_ocr'
    )
    parser.add_argument(
        "--base_url", type=str, default="http://127.0.0.1:8000",
        help="Base URL for vLLM server (e.g., http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--model_name", type=str, default="model",
        help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help=""
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help=""
    )
    parser.add_argument(
        "--max_completion_tokens", type=int, default=16384,
        help=""
    )
    parser.add_argument(
        "--num_thread", type=int, default=16,
        help=""
    )
    # parser.add_argument(
    #     "--fitz_preprocess", type=bool, default=False,
    #     help="False will use tikz dpi upsample pipeline, good for images which has been render with low dpi, but maybe result in higher computational costs"
    # )
    parser.add_argument(
        "--min_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help=""
    )
    # backend options
    parser.add_argument(
        "--backend", type=str, choices=["vllm", "replicate"], default="vllm",
        help="backend to use for inference"
    )
    parser.add_argument(
        "--replicate_deployment", type=str, default=None,
        help="replicate deployment name, e.g., owner/name; if not set, uses public model sljeff/dots.ocr"
    )
    parser.add_argument(
        "--api_token", type=str, default=None,
        help="API token for the selected backend (required for both vllm and replicate)"
    )

    args = parser.parse_args()

    dots_ocr_parser = DotsOCRParser(
        backend=args.backend,
        base_url=args.base_url,
        api_token=args.api_token,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        replicate_deployment=args.replicate_deployment,
    )

    result = dots_ocr_parser.parse_file(
        args.input_path, 
        prompt_mode=args.prompt,
        bbox=args.bbox,
        )
    


if __name__ == "__main__":
    main()
