#!/usr/bin/env python3

import yaml
import argparse
from pathlib import Path

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    parser = argparse.ArgumentParser(description='Export config variables to shell format')
    parser.add_argument('--shell', action='store_true', help='Output in shell format')
    args = parser.parse_args()

    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Flatten the nested dictionary
    flat_config = flatten_dict(config)

    # Output in shell format
    if args.shell:
        for key, value in flat_config.items():
            # Convert value to string, handling special characters
            if isinstance(value, str):
                value = f'"{value}"'  # Wrap strings in quotes
            else:
                value = str(value)
            print(f'export {key}={value}')

if __name__ == '__main__':
    main() 