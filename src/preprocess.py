from datasets import load_dataset

def transform_json(record):
    system = '''You are a model that converts input JSON structures into a different JSON format based on specific rules. The JSON provides information for a newspaper distribution tour, including one or more areas that describe the geographical location of the tour based on zip codes and districts, as well as the occupancy units (micro zip codes) that correspond to a zip code. Note that the first five digits of an occupancy unit represent the zip code where it is located.'''
    
    instruction = "Select the occupancy units that best describe the tour using the 'location', 'district', 'districtHousholdsProportion', 'copies', and 'netHousehold' information from the following JSON:"

    json = "{ 'tour': " + f'{record["tour"]}, ' + "'areas:' " + f'{record["areas"]}, ' + "'occupancyUnits:' " + f'{record["occupancyUnits"]} ' + "}"

    answer = f'{record["actualOccupancyUnits"]}'

    text = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>{instruction}\n\n{json}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n{answer}<|eot_id|>'

    return { "text": text }

def transform(path: str):
    dataset = load_dataset("json", data_files=path)
    transformed_dataset = dataset.map(transform_json)
    return transformed_dataset.remove_columns(column_names=['tour', 'occupancyUnits', 'areas', 'actualOccupancyUnits'])

if __name__ == "__main__":
    transform("data/dataset_2024-08-22T15-44-58-752Z.json");
