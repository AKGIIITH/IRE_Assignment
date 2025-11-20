import pandas as pd
import jellyfish
import networkx as nx
import itertools
import re

# 1. Load Data
df = pd.read_csv('dedup_data.csv')

original_count = len(df)
df = df.drop_duplicates(subset=['id'], keep='first')
print(f"Loaded {len(df)} rows. Dropped {original_count - len(df)} duplicate ID rows.")

# 2. Standardize Data
std_df = df.copy()

# Standardize soc_sec_id, postcode, street_number
for col in ['soc_sec_id', 'postcode', 'street_number']:
    std_df[col] = std_df[col].astype(str).str.replace(' ', '').str.replace('-', '')
    std_df[col] = std_df[col].replace('nan', '')

# Standardize address fields
for col in ['address_1', 'address_2']:
    std_df[col] = std_df[col].fillna('').astype(str).str.lower()
    std_df[col] = std_df[col].apply(lambda x: re.sub(r'[^\w]', '', x))

# Standardize name fields
std_df['given_name'] = std_df['given_name'].fillna('').astype(str).str.lower()
std_df['surname'] = std_df['surname'].fillna('').astype(str).str.lower()

# Handle swapped fields
std_df['name_set'] = std_df.apply(
    lambda row: frozenset([row['given_name'], row['surname']]) if row['given_name'] or row['surname'] else frozenset(),
    axis=1
)
std_df['address_set'] = std_df.apply(
    lambda row: frozenset([row['address_1'], row['address_2']]) if row['address_1'] or row['address_2'] else frozenset(),
    axis=1
)

# Create blocking key
std_df['surname_soundex'] = std_df['surname'].apply(
    lambda x: jellyfish.soundex(x) if x else ''
)

# Standardize date_of_birth
std_df['date_of_birth'] = std_df['date_of_birth'].fillna('').astype(str)

# 3. Blocking (Candidate Generation)
candidate_pairs = set()

# Block 1: Phonetic Surname
for soundex, group in std_df.groupby('surname_soundex'):
    if soundex and len(group) > 1:
        pairs = itertools.combinations(group['id'].tolist(), 2)
        candidate_pairs.update(pairs)

# Block 2: Postcode
for postcode, group in std_df.groupby('postcode'):
    if postcode and len(group) > 1:
        pairs = itertools.combinations(group['id'].tolist(), 2)
        candidate_pairs.update(pairs)

# Block 3: Date of Birth
for dob, group in std_df.groupby('date_of_birth'):
    if dob and len(group) > 1:
        pairs = itertools.combinations(group['id'].tolist(), 2)
        candidate_pairs.update(pairs)

print(f"Total candidate pairs: {len(candidate_pairs)}")

# 4. Comparison and Scoring
data_dict = std_df.set_index('id').to_dict('index')

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 0.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def calculate_similarity(id1, id2, data_dict):
    rec1 = data_dict[id1]
    rec2 = data_dict[id2]
    
    sim_name = jaccard_similarity(rec1['name_set'], rec2['name_set'])
    sim_addr = jaccard_similarity(rec1['address_set'], rec2['address_set'])
    
    sim_ssn = 0.0
    if rec1['soc_sec_id'] and rec2['soc_sec_id']:
        sim_ssn = jellyfish.jaro_winkler_similarity(rec1['soc_sec_id'], rec2['soc_sec_id'])
    
    sim_dob = 0.0
    if rec1['date_of_birth'] and rec2['date_of_birth']:
        sim_dob = jellyfish.jaro_winkler_similarity(rec1['date_of_birth'], rec2['date_of_birth'])
    
    score = (0.4 * sim_ssn) + (0.25 * sim_name) + (0.2 * sim_dob) + (0.15 * sim_addr)
    return score

# 5. Clustering (Grouping)
G = nx.Graph()
G.add_nodes_from(df['id'].tolist())

MATCH_THRESHOLD = 0.85

for id1, id2 in candidate_pairs:
    score = calculate_similarity(id1, id2, data_dict)
    if score > MATCH_THRESHOLD:
        G.add_edge(id1, id2)

# Find connected components
final_groups = [list(component) for component in nx.connected_components(G)]


# 6. Output Results (Console output for verification)
print(f"\nTotal unique persons: {len(final_groups)}")

# Get largest groups (more than 1 id)
large_groups = [g for g in final_groups if len(g) > 1]
large_groups.sort(key=len, reverse=True)

print(f"\nTop 20 largest groups:")
for i, group in enumerate(large_groups[:20], 1):
    print(f"Group {i}: {len(group)} ids - {group}")

# 7. Generate Submission File

print("\nGenerating 'prediction.csv'...")

submission_data = []

# Iterate through all groups found by the clustering algorithm
for idx, group_nodes in enumerate(final_groups):
    # Create the group identifier (e.g., group_0, group_1)
    current_group_id = f"group_{idx}"
    
    # Add every ID in this group to the list
    for record_id in group_nodes:
        record_ID=f"id{record_id}"
        submission_data.append({
            'id': record_ID,
            'group_id': current_group_id
        })

# Create the DataFrame
submission_df = pd.DataFrame(submission_data)

# Sort by ID or Group ID to keep it tidy
submission_df = submission_df.sort_values(by=['group_id', 'id'])

# Save to CSV without the index
submission_df.to_csv('prediction.csv', index=False)

print(f"Successfully saved {len(submission_df)} rows to 'prediction.csv'.")