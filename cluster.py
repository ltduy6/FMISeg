import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import pandas as pd
import os

def cluster_prompts(prompts, url="microsoft/BiomedVLP-CXR-BERT-specialized", num_prototypes=16, save_path="./prototypes"):
    tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    model = AutoModel.from_pretrained(url, trust_remote_code=True)
    embedding_list = []
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            batch_prompts = prompts[i:i+32]
            tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_prompts,
                                               add_special_tokens=True,
                                               padding='longest',
                                               return_tensors='pt')
            batch_embeddings = model.get_projected_text_embeddings(input_ids=tokenizer_output.input_ids,
                                                 attention_mask=tokenizer_output.attention_mask)
            embedding_list.append(batch_embeddings)
    
    embeddings = torch.cat(embedding_list, dim=0)
    embeddings_np = embeddings.numpy()

    print(f"Extracted {embeddings_np.shape[0]} embeddings of dimension {embeddings_np.shape[1]}")

    print(f"Clustering into {num_prototypes} prototypes...")
    kmeans = KMeans(n_clusters=num_prototypes, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings_np)

    silhouette_avg = silhouette_score(embeddings_np, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg}")

    # Create prototype matrix
    prototype_centers = torch.from_numpy(kmeans.cluster_centers_).float()  # Shape: [num_prototypes, embedding_dim]
    
    # Analyze cluster distribution and print texts in each cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    unique_prompts = set()
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS")
    print("="*80)
    
    for cluster_id, count in zip(unique, counts):
        print(f"\n--- CLUSTER {cluster_id} ({count} samples) ---")
        
        # Get indices of samples in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        # Print first 10 texts in this cluster (or all if less than 10)
        max_display = min(10, len(cluster_indices))
        for i, idx in enumerate(cluster_indices[:max_display]):
            print(f"  {i+1:2d}. {prompts[idx]}")
            unique_prompts.add(prompts[idx])

        if len(cluster_indices) > max_display:
            print(f"  ... and {len(cluster_indices) - max_display} more samples")
        print()
    
    list_unique_prompts = list(unique_prompts)
    tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=list_unique_prompts,
                                               add_special_tokens=True,
                                               padding='longest',
                                               return_tensors='pt')
    batch_embeddings = model.get_projected_text_embeddings(input_ids=tokenizer_output.input_ids,
                                                 attention_mask=tokenizer_output.attention_mask)
    sim = torch.mm(embeddings, embeddings.t())
    print("Similarity matrix:", sim)

    # Create prototype space dictionary
    prototype_space = {
        'prototypes': prototype_centers,
        'cluster_labels': cluster_labels,
        'prompts': prompts,
        'embeddings': embeddings,
        'num_prototypes': num_prototypes,
        'silhouette_score': silhouette_avg,
        'embedding_dim': embeddings.shape[1]
    }
    
    # Analyze cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("Cluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} samples")
    
    # Save prototype space
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"prototype_space_{num_prototypes}.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(prototype_space, f)
    print(f"Prototype space saved to: {save_file}")
    
    return prototype_space

def optimize_num_prototypes(prompts, url="microsoft/BiomedVLP-CXR-BERT-specialized", 
                          min_k=8, max_k=32, save_path="./prototypes"):
    """
    Find optimal number of prototypes using silhouette analysis.
    
    Args:
        prompts (list): List of text prompts
        url (str): Model URL
        min_k, max_k (int): Range of prototype numbers to test
        save_path (str): Save path for results
    
    Returns:
        dict: Results for different k values
    """
    # Extract embeddings once
    tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    model = AutoModel.from_pretrained(url, trust_remote_code=True)
    
    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            batch_prompts = prompts[i:i+32]
            tokenizer_output = tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=batch_prompts,
                add_special_tokens=True,
                padding='longest',
                return_tensors='pt'
            )
            batch_embeddings = model.get_projected_text_embeddings(
                input_ids=tokenizer_output.input_ids,
                attention_mask=tokenizer_output.attention_mask
            )
            embeddings_list.append(batch_embeddings.cpu())
    
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings_np = embeddings.numpy()
    
    # Test different k values
    results = {}
    best_k = min_k
    best_score = -1
    
    for k in range(min_k, max_k + 1):
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_np)
        score = silhouette_score(embeddings_np, labels)
        
        results[k] = {
            'silhouette_score': score,
            'centers': torch.from_numpy(kmeans.cluster_centers_).float(),
            'labels': labels
        }
        
        if score > best_score:
            best_score = score
            best_k = k
        
        print(f"  k={k}: Silhouette Score = {score:.4f}")
    
    print(f"Best k: {best_k} with score: {best_score:.4f}")
    
    # Save optimization results
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "prototype_optimization.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    return results, best_k

def load_prototype_space(save_path="./prototypes", num_prototypes=16):
    """Load saved prototype space."""
    save_file = os.path.join(save_path, f"prototype_space_{num_prototypes}.pkl")
    with open(save_file, 'rb') as f:
        return pickle.load(f)

# Example usage:
if __name__ == "__main__":
    file_path = "./data/MosMedData+/prompt/train.csv"
    df = pd.read_csv(file_path)
    prompts = df['Description'].tolist()
    # get the last part of prompts after splitting by ','
    prompts = [prompt.split(',')[-1].strip() for prompt in prompts]
    
    # Create prototype space
    prototype_space = cluster_prompts(prompts, num_prototypes=22)