tuned_model_path = "/Users/philipprollmann/Dev/chainmatics/llm/ai-digitizer/ggml-model-Q4_K_M.gguf"
sys_message = "You are a model that converts input JSON structures into a different JSON format based on specific rules. The JSON provides information for a newspaper distribution tour, including one or more areas that describe the geographical location of the tour based on zip codes and districts, as well as the occupancy units (micro zip codes) that correspond to a zip code. Note that the first five digits of an occupancy unit represent the zip code where it is located."

cmds = []

base_model = f"FROM {tuned_model_path}"

template = '''TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"
"""'''

params = '''PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token"'''

system = f'''SYSTEM """{sys_message}"""'''

cmds.append(base_model)
cmds.append(template)
cmds.append(params)
cmds.append(system)

def generate_modelfile(cmds):
    content = ""
    for command in cmds:
        content += command + "\n"
    print(content)
    with open("Modelfile", "w") as file:
        file.write(content)


generate_modelfile(cmds)