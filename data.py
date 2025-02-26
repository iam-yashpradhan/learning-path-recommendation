import json
import pandas as pd
import ast

data = pd.read_csv('response_data_with_roles.csv')

# Print the contents of the 'choices' column to debug
def extract_value(row_str, key):
    try:
        parsed_list = ast.literal_eval(row_str)
        return parsed_list[0]['message']['parsed'].get(key, None)
    except Exception:
        return None

# data['blog_title'] = data['choices'].apply(lambda x: extract_value(x, 'blog_title'))
# data['category'] = data['choices'].apply(lambda x: extract_value(x, 'category'))
# data['roles'] = data['choices'].apply(lambda x: extract_value(x, 'roles'))
# data.to_csv('entity_dataset.csv', index=False)
# print(data)

def add_a_column(data, column_name, key):
    data[column_name] = key
    return data

data = add_a_column(data, 'category', 'learning-path')
data.to_csv('updated_entity_dataset.csv', index=False)

