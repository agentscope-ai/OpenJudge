import json

file1_path = '../../bfcl_v3/tool_call/tool_call_bfcl_v3_multiple_eval_data.jsonl'
file2_path = '../../bfcl_v3/tool_call/tool_call_bfcl_v3_multiple_raw_llm_output.jsonl'
output_path = '../../bfcl_v3/tool_call/tool_call_bfcl_v3_multiple_eval_data_new_cases.jsonl'
DEFAULT_MAX_SCORE = 3

with open(file1_path, 'r', encoding='utf-8') as f1, \
     open(file2_path, 'r', encoding='utf-8') as f2, \
     open(output_path, 'w', encoding='utf-8') as out:

    for line1, line2 in zip(f1, f2):
        data1 = json.loads(line1.strip())
        data2 = json.loads(line2.strip())

        name1 = data1['parameters']['tool_calls'][0]['function']['name']
        name2 = data2['name']

        if name1 != name2:
            data1['parameters']['tool_calls'][0]['function']['name'] = name2
            if "id" in data1["parameters"]["tool_calls"][0]:
                del data1["parameters"]["tool_calls"][0]["id"]
            data1.pop('min_expect_score', None)
            # validate max_expect_score field and adjust value manually in output file
            data1['max_expect_score'] = DEFAULT_MAX_SCORE
            out.write(json.dumps(data1, ensure_ascii=False) + '\n')
