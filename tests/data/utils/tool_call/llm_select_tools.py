import json
import re
from dashscope import Generation


def simplify_params_for_prompt(param_schema):
    props = param_schema.get("properties", {})
    required = set(param_schema.get("required", []))
    desc_lines = []
    for name, info in props.items():
        ptype = info.get("type", "unknown")
        is_required = name in required
        opt_text = " (required)" if is_required else " (optional, default: " + str(info.get("default", "N/A")) + ")"
        desc_lines.append(f"  - {name}: {ptype}{opt_text}")
    return "\n".join(desc_lines) if desc_lines else "  (no parameters)"


def build_prompt(content: str, functions: list) -> str:
    prompt = (
        "You are a precise function router. Given a user question and a list of available functions, "
        "select exactly one function that can best answer the question.\n\n"
        "Respond ONLY with a JSON object containing two keys:\n"
        '- "name": the function name as a string,\n'
        '- "arguments": a JSON STRING (not object) containing only the required parameter values inferred from the question.\n'
        "Do NOT include optional parameters that use default values unless explicitly mentioned.\n"
        "Do NOT output any other text, explanation, or markdown.\n\n"
        f"User question: \"{content}\"\n\n"
        "Available functions:\n"
    )
    for i, func in enumerate(functions, 1):
        name = func["name"]
        desc = func.get("description", "No description.")
        params_desc = simplify_params_for_prompt(func["parameters"])
        prompt += f"{i}. Function name: `{name}`\n   Description: {desc}\n   Parameters:\n{params_desc}\n\n"
    prompt += "Your response:"
    return prompt


def extract_and_stringify_arguments(text: str) -> dict:
    text = text.strip()
    outer_obj = None
    try:
        outer_obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                outer_obj = json.loads(match.group(1))
            except json.JSONDecodeError:
                print(f"Found invalid json format in pattern match: {match}")

    if outer_obj and isinstance(outer_obj, dict):
        name = outer_obj.get("name")
        args = outer_obj.get("arguments", {})

        if isinstance(args, str):
            try:
                json.loads(args)
                arguments_str = args
            except json.JSONDecodeError:
                arguments_str = "{}"
                print(f"Found invalid json format in arguments, set to {arguments_str}")
        elif isinstance(args, dict):
            arguments_str = json.dumps(args)
        else:
            arguments_str = "{}"

        return {"name": name, "arguments": arguments_str}

    return {"name": "", "arguments": "{}"}


def process_jsonl(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                content = record["question"][0][0]["content"]
                functions = record["function"]
                prompt = build_prompt(content, functions)
                response = Generation.call(
                    model="qwen-max",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )

                reply = response.output.text
                result = extract_and_stringify_arguments(reply)
                result = json.dumps(result, ensure_ascii=False)
                print(f"id {line_num}, result: {result}")
                fout.write(result + "\n")
                fout.flush()

            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                fout.write('{"name": null, "arguments": "{}"}\n')


if __name__ == "__main__":
    process_jsonl("../../bfcl_v3/tool_call/tool_call_bfcl_v3_multiple_raw.jsonl",
                 "../../bfcl_v3/tool_call/tool_call_bfcl_v3_multiple_raw_llm_output.jsonl")
