import os
import glob
import pickle
import numpy as np
import umap

TOP_ELEMENT = 3072

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


#RESULT_DIR = '/Volumes/MDatahubDev/analysis_20240413/result'
RESULT_DIR = '/Volumes/MDatahubDev/Total_result'

ofp = open('./project_keyfindings.txt', 'w')

print('Loading data...')
TARGET_PMCs = [os.path.basename(pmcdir) for pmcdir in glob.glob(os.path.join(RESULT_DIR, 'PMC*'))]
ALL_EMBEDDINGS = []

for i, TARGET_PMC in enumerate(TARGET_PMCs):
    embedding_file = os.path.join(RESULT_DIR, TARGET_PMC, f'{TARGET_PMC}_project_embedding.pkl')
    if not os.path.exists(embedding_file):
        print(f'No project embedding file: {embedding_file}')
        continue
    with open(embedding_file, 'rb') as f:
        project_embedding = pickle.load(f)

    vec = project_embedding['embedding']
    vec = vec[:TOP_ELEMENT]
    vec = normalize_l2(vec)
    
    ofp.write(f'{TARGET_PMC}: {project_embedding["key_findings"]} \n')
    ALL_EMBEDDINGS.append(vec)

# 
ALL_EMBEDDINGS = np.vstack(ALL_EMBEDDINGS)
print(ALL_EMBEDDINGS.shape)

model = umap.UMAP(verbose=True,
                  n_neighbors=3,
                  min_dist=0.1,
                  n_components=2,
                  metric='cosine')
result = model.fit_transform(ALL_EMBEDDINGS)

# Save first 2 components
with open('./project_umap.pkl', 'wb') as f:
    pickle.dump(result, f)



