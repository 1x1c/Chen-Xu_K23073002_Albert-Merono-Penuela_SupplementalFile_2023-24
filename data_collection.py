import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Wikidata's SPARQL queries and API endpoints
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
API_ENDPOINT = "https://www.wikidata.org/w/api.php"

def get_edit_data(item_type: str, limit: int = 10) -> pd.DataFrame:
    item_id = 18 if item_type == "image" else (51 if item_type == "audio" else 10)
    # SPARQL query to get Wikidata entries containing images
    SPARQL_QUERY = f"""
        SELECT ?item ?itemLabel ?{item_type}
        WHERE {{
            ?item wdt:P{item_id} ?{item_type}.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        LIMIT {limit}
    """

    # Sending SPARQL queries to Wikidata
    response = requests.get(SPARQL_ENDPOINT, params={'query': SPARQL_QUERY, 'format': 'json'})
    data = response.json()

    #Parsing query results
    items = []
    for item in tqdm(data['results']['bindings'], desc=item_type):
        items.append({
            'item': item['item']['value'],
            'label': item['itemLabel']['value'],
            item_type: item[item_type]['value']
        })

    # Convert the result into a DataFrame
    df_items = pd.DataFrame(items)
    return df_items

# Function to obtain edit history data in parallel
def fetch_edit_history(item_url: str) -> list[dict]:
    item_id = item_url.split('/')[-1]  # 从URL中提取Q号
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'revisions',
        'titles': item_id,
        'rvprop': 'user|timestamp',
        'rvlimit': '500'
    }
    try:
        response = requests.get(API_ENDPOINT, params=params)
        revisions = response.json().get('query', {}).get('pages', {})

        editor_data = []
        for page_id in revisions:
            page_info = revisions[page_id]
            if 'revisions' in page_info:
                for rev in page_info['revisions']:
                    # Use .get() to safely get the 'user' key
                    editor = rev.get('user', 'anonymous')  # 如果'user'不存在，返回'anonymous'
                    # Add data only if the editor is not anonymous
                    if editor != 'anonymous':
                        editor_data.append({'editor': editor, 'item': item_id, 'timestamp': rev.get('timestamp', None)})
    except Exception as e:
        print(e)
        editor_data = [{'editor': None, 'item': item_id, 'timestamp': None}]
    return editor_data

def get_editors(df_items: pd.DataFrame) -> pd.DataFrame:
    editors_data = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_item = {executor.submit(fetch_edit_history, item): item for item in df_items['item']}
        for future in as_completed(future_to_item):
            editors_data.extend(future.result())

    #Convert editor data to DataFrame
    df_editors = pd.DataFrame(editors_data)
    
    return df_editors

def get_editors_by_type(item_type: str, data_dir: str, num_split: int = 10) -> pd.DataFrame:
    items = pd.read_csv(os.path.join(data_dir, "items", f"{item_type}_items.csv"))
    size = int(len(items) / num_split) + 1
    dfs = []
    for i in tqdm(range(num_split)):
        dfs.append(get_editors(items.iloc[i * size: (i+1) * size]))
        editors_df = pd.concat(dfs)
    return editors_df
        
        

if __name__ == "__main__":
    DATA_DIR = "data"
    image_editors_df = get_editors_by_type("image", DATA_DIR)
    image_editors_df.to_csv(os.path.join(DATA_DIR, "multi-media_editors", ("image_editors_wt.csv")))

    video_editors_df = get_editors_by_type("video", DATA_DIR)
    video_editors_df.to_csv(os.path.join(DATA_DIR, "multi-media_editors", ("video_editors_wt.csv")))

    audio_editors_df = get_editors_by_type("audio", DATA_DIR)
    audio_editors_df.to_csv(os.path.join(DATA_DIR, "multi-media_editors", ("audio_editors_wt.csv")))

