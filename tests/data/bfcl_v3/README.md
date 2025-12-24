# 说明文档
本文档包含多个数据文件和Python 脚本，用于工具调用评估数据的生成和处理。

## JSONL 数据文件

### 1. `tool_call_bfcl_v3_multiple_raw.jsonl`
- **类型**: bfcl_v3_multiple原始输入数据
- **格式**: 每行一个JSON对象
- **示例**:
```
{"id": "multiple_0", "question": [[{"role": "user", "content": "Can I find the dimensions and properties of a triangle, if I know its three sides are 5 units, 4 units and 3 units long?"}]], "function": [{"name": "triangle_properties.get", "description": "Retrieve the dimensions, such as area and perimeter, of a triangle if lengths of three sides are given.", "parameters": {"type": "dict", "properties": {"side1": {"type": "integer", "description": "The length of first side of the triangle."}, "side2": {"type": "integer", "description": "The length of second side of the triangle."}, "side3": {"type": "integer", "description": "The length of third side of the triangle."}, "get_area": {"type": "boolean", "description": "A flag to determine whether to calculate the area of triangle. Default is true.", "default": true, "optional": true}, "get_perimeter": {"type": "boolean", "description": "A flag to determine whether to calculate the perimeter of triangle. Default is true.", "default": true, "optional": true}, "get_angles": {"type": "boolean", "description": "A flag to determine whether to calculate the internal angles of triangle. Default is true.", "default": true, "optional": true}}, "required": ["side1", "side2", "side3"]}}, {"name": "circle_properties.get", "description": "Retrieve the dimensions, such as area and circumference, of a circle if radius is given.", "parameters": {"type": "dict", "properties": {"radius": {"type": "float", "description": "The length of radius of the circle."}, "get_area": {"type": "boolean", "description": "A flag to determine whether to calculate the area of circle. Default is true.", "default": true, "optional": true}, "get_circumference": {"type": "boolean", "description": "A flag to determine whether to calculate the circumference of circle. Default is true.", "default": true, "optional": true}}, "required": ["radius"]}}]}
{"id": "multiple_1", "question": [[{"role": "user", "content": "Calculate the area of a triangle, given the lengths of its three sides: 3, 4, and 5."}]], "function": [{"name": "math.triangle_area_heron", "description": "Calculates the area of a triangle using Heron's formula, given the lengths of its three sides.", "parameters": {"type": "dict", "properties": {"side1": {"type": "integer", "description": "Length of the first side of the triangle."}, "side2": {"type": "integer", "description": "Length of the second side of the triangle."}, "side3": {"type": "integer", "description": "Length of the third side of the triangle."}}, "required": ["side1", "side2", "side3"]}}, {"name": "math.circle_area", "description": "Calculates the area of a circle given its radius.", "parameters": {"type": "dict", "properties": {"radius": {"type": "float", "description": "The radius of the circle."}}, "required": ["radius"]}}, {"name": "math.triangle_area_base_height", "description": "Calculates the area of a triangle using the formula (1/2)base*height.", "parameters": {"type": "dict", "properties": {"base": {"type": "float", "description": "The base length of the triangle."}, "height": {"type": "float", "description": "The height of the triangle."}}, "required": ["base", "height"]}}]}
```

### 2. `tool_call_bfcl_v3_multiple_gt.jsonl`
- **类型**: bfcl_v3_multiple ground-truth数据
- **格式**: 每行一个JSON对象
- **示例**:
```
{"id": "multiple_0", "ground_truth": [{"triangle_properties.get": {"side1": [5], "side2": [4], "side3": [3], "get_area": ["", true], "get_perimeter": ["", true], "get_angles": ["", true]}}]}
{"id": "multiple_1", "ground_truth": [{"math.triangle_area_heron": {"side1": [3], "side2": [4], "side3": [5]}}]}
```

### 3. `tool_call_bfcl_v3_multiple_multiple.jsonl`
- **类型**: bfcl_v3_multiple的raw和gt两个文件生成的数据
- **格式**: 每行一个JSON对象
- **示例**:
```
{"id": "multiple_0", "question": [[{"role": "user", "content": "Can I find the dimensions and properties of a triangle, if I know its three sides are 5 units, 4 units and 3 units long?"}]], "function": [{"name": "triangle_properties.get", "description": "Retrieve the dimensions, such as area and perimeter, of a triangle if lengths of three sides are given.", "parameters": {"type": "dict", "properties": {"side1": {"type": "integer", "description": "The length of first side of the triangle."}, "side2": {"type": "integer", "description": "The length of second side of the triangle."}, "side3": {"type": "integer", "description": "The length of third side of the triangle."}, "get_area": {"type": "boolean", "description": "A flag to determine whether to calculate the area of triangle. Default is true.", "default": true, "optional": true}, "get_perimeter": {"type": "boolean", "description": "A flag to determine whether to calculate the perimeter of triangle. Default is true.", "default": true, "optional": true}, "get_angles": {"type": "boolean", "description": "A flag to determine whether to calculate the internal angles of triangle. Default is true.", "default": true, "optional": true}}, "required": ["side1", "side2", "side3"]}}, {"name": "circle_properties.get", "description": "Retrieve the dimensions, such as area and circumference, of a circle if radius is given.", "parameters": {"type": "dict", "properties": {"radius": {"type": "float", "description": "The length of radius of the circle."}, "get_area": {"type": "boolean", "description": "A flag to determine whether to calculate the area of circle. Default is true.", "default": true, "optional": true}, "get_circumference": {"type": "boolean", "description": "A flag to determine whether to calculate the circumference of circle. Default is true.", "default": true, "optional": true}}, "required": ["radius"]}}], "ground_truth": [{"triangle_properties.get": {"side1": [5], "side2": [4], "side3": [3], "get_area": ["", true], "get_perimeter": ["", true], "get_angles": ["", true]}}]}
{"id": "multiple_1", "question": [[{"role": "user", "content": "Calculate the area of a triangle, given the lengths of its three sides: 3, 4, and 5."}]], "function": [{"name": "math.triangle_area_heron", "description": "Calculates the area of a triangle using Heron's formula, given the lengths of its three sides.", "parameters": {"type": "dict", "properties": {"side1": {"type": "integer", "description": "Length of the first side of the triangle."}, "side2": {"type": "integer", "description": "Length of the second side of the triangle."}, "side3": {"type": "integer", "description": "Length of the third side of the triangle."}}, "required": ["side1", "side2", "side3"]}}, {"name": "math.circle_area", "description": "Calculates the area of a circle given its radius.", "parameters": {"type": "dict", "properties": {"radius": {"type": "float", "description": "The radius of the circle."}}, "required": ["radius"]}}, {"name": "math.triangle_area_base_height", "description": "Calculates the area of a triangle using the formula (1/2)base*height.", "parameters": {"type": "dict", "properties": {"base": {"type": "float", "description": "The base length of the triangle."}, "height": {"type": "float", "description": "The height of the triangle."}}, "required": ["base", "height"]}}], "ground_truth": [{"math.triangle_area_heron": {"side1": [3], "side2": [4], "side3": [5]}}]}
```

### 5. `tool_call_bfcl_v3_multiple_eval_data.jsonl`
- **类型**: bfcl_v3_multiple处理后的数据, 用于跑grader评估测试
- **格式**: 每行一个JSON对象
- **示例**:
```
{"index": 0, "parameters": {"query": [{"message": {"role": "user", "content": "Can I find the dimensions and properties of a triangle, if I know its three sides are 5 units, 4 units and 3 units long?"}}], "tool_calls": [{"function": {"name": "triangle_properties.get", "arguments": "{\"side1\": 5, \"side2\": 4, \"side3\": 3, \"get_area\": true, \"get_perimeter\": true, \"get_angles\": true}"}, "id": "call_triangle_properties.get_06c54c"}], "tool_definitions": [{"type": "function", "function": {"name": "triangle_properties.get", "description": "Retrieve the dimensions, such as area and perimeter, of a triangle if lengths of three sides are given.", "parameters": {"type": "dict", "properties": {"side1": {"type": "integer", "description": "The length of first side of the triangle."}, "side2": {"type": "integer", "description": "The length of second side of the triangle."}, "side3": {"type": "integer", "description": "The length of third side of the triangle."}, "get_area": {"type": "boolean", "description": "A flag to determine whether to calculate the area of triangle. Default is true.", "default": true, "optional": true}, "get_perimeter": {"type": "boolean", "description": "A flag to determine whether to calculate the perimeter of triangle. Default is true.", "default": true, "optional": true}, "get_angles": {"type": "boolean", "description": "A flag to determine whether to calculate the internal angles of triangle. Default is true.", "default": true, "optional": true}}, "required": ["side1", "side2", "side3"]}}}, {"type": "function", "function": {"name": "circle_properties.get", "description": "Retrieve the dimensions, such as area and circumference, of a circle if radius is given.", "parameters": {"type": "dict", "properties": {"radius": {"type": "float", "description": "The length of radius of the circle."}, "get_area": {"type": "boolean", "description": "A flag to determine whether to calculate the area of circle. Default is true.", "default": true, "optional": true}, "get_circumference": {"type": "boolean", "description": "A flag to determine whether to calculate the circumference of circle. Default is true.", "default": true, "optional": true}}, "required": ["radius"]}}}]}, "min_expect_score": 4}
{"index": 1, "parameters": {"query": [{"message": {"role": "user", "content": "Calculate the area of a triangle, given the lengths of its three sides: 3, 4, and 5."}}], "tool_calls": [{"function": {"name": "math.triangle_area_heron", "arguments": "{\"side1\": 3, \"side2\": 4, \"side3\": 5}"}, "id": "call_math.triangle_area_heron_045dc9"}], "tool_definitions": [{"type": "function", "function": {"name": "math.triangle_area_heron", "description": "Calculates the area of a triangle using Heron's formula, given the lengths of its three sides.", "parameters": {"type": "dict", "properties": {"side1": {"type": "integer", "description": "Length of the first side of the triangle."}, "side2": {"type": "integer", "description": "Length of the second side of the triangle."}, "side3": {"type": "integer", "description": "Length of the third side of the triangle."}}, "required": ["side1", "side2", "side3"]}}}, {"type": "function", "function": {"name": "math.circle_area", "description": "Calculates the area of a circle given its radius.", "parameters": {"type": "dict", "properties": {"radius": {"type": "float", "description": "The radius of the circle."}}, "required": ["radius"]}}}, {"type": "function", "function": {"name": "math.triangle_area_base_height", "description": "Calculates the area of a triangle using the formula (1/2)base*height.", "parameters": {"type": "dict", "properties": {"base": {"type": "float", "description": "The base length of the triangle."}, "height": {"type": "float", "description": "The height of the triangle."}}, "required": ["base", "height"]}}}]}, "min_expect_score": 4}
```

## python脚本

### 1. `generate_bfcl_tool_call_data.py`
将`tool_call_bfcl_v3_multiple_raw.jsonl`和`tool_call_bfcl_v3_multiple_gt.jsonl`两个文件合并生成`tool_call_bfcl_v3_multiple_multiple.jsonl`
```
python generate_bfcl_tool_call_data.py <tool_call_bfcl_v3_multiple_raw.jsonl> <tool_call_bfcl_v3_multiple_gt.jsonl> <tool_call_bfcl_v3_multiple.jsonl>
```

### 2. `process_bfcl_tool_call_data.py`
处理`tool_call_bfcl_v3_multiple_multiple.jsonl`，用于跑grader评估测试
```
python process_bfcl_tool_call_data.py --input_path <tool_call_bfcl_v3_multiple.jsonl> --output_path <tool_call_bfcl_v3_multiple_eval_data.jsonl> --min_expect_score 4
```

### 3. `run_grader_eval_bfcl_dataset.py`
用于评估指定评分器（例如 `ToolCallAccuracyGrader`）在 BFCL 数据集的一组测试用例(`tool_call_bfcl_v3_multiple_eval_data.jsonl`)上的性能。
```
python run_grader_eval_bfcl_dataset.py --grader_name ToolCallAccuracyGrader --sample_size 10 --skip 1
```

### 4. `llm_select_tools.py`
用llm选择工具，结果将用于对比bfcl_v3 ground-truth
```
python llm_select_tools.py
```

### 6. `generate_new_cases.py`
基于llm选择工具的结果和bfcl_v3 ground-truth做比较，如果有不同，将用前者代替后者。
注：需要人工判断一下每个case的max_expect_score（默认值=3）是否合理，并做适当调整
```
python generate_new_cases.py
```
再按步骤3运行测试脚本
```
python run_grader_eval_bfcl_dataset.py --case_file <tool_call_bfcl_v3_multiple_eval_data_new_cases.jsonl> --grader_name ToolCallAccuracyGrader 
```