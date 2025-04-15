import os
import json
import csv

data_directory = 'data/jsons/'
output_csv = 'output.csv'

extracted_data = []

for filename in os.listdir(data_directory):
    if filename.endswith('.json'):  
        file_path = os.path.join(data_directory, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
           
                source = data.get('source', 'N/A')  
                content = data.get('content', 'N/A')  
                article_id = data.get('ID', 'N/A') 
                bias = data.get('bias', 'N/A') 
        
                extracted_data.append((source, content, article_id, bias))
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {filename}: {e}")

with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write the header
    csv_writer.writerow(['source', 'content', 'ID', 'bias'])
    
    # Write each row of extracted data
    csv_writer.writerows(extracted_data)

print(f"CSV file '{output_csv}' created successfully with {len(extracted_data)} records.")
