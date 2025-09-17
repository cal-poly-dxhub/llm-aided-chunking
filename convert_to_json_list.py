#!/usr/bin/env python3

import json
import sys
import re

def convert_to_json_list(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read().strip()
    
    # Split on }{ pattern to separate JSON objects
    json_strings = re.split(r'}\s*{', content)
    
    # Fix the split by adding back the braces
    if len(json_strings) > 1:
        json_strings[0] += '}'
        json_strings[-1] = '{' + json_strings[-1]
        for i in range(1, len(json_strings) - 1):
            json_strings[i] = '{' + json_strings[i] + '}'
    
    # Parse and collect JSON objects
    json_objects = []
    for json_str in json_strings:
        try:
            obj = json.loads(json_str)
            json_objects.append(obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            continue
    
    # Write as JSON array
    with open(output_file, 'w') as f:
        json.dump(json_objects, f, indent=2)
    
    print(f"Converted {len(json_objects)} objects to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_json_list.py <input_file> <output_file>")
        sys.exit(1)
    
    convert_to_json_list(sys.argv[1], sys.argv[2])
