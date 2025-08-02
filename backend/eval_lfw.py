import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from timm.models import create_model
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
from spikingjelly.activation_based import functional
import models.spikingresformer

class LFWPairsDataset(Dataset):
    """LFW dataset that loads pairs from standard protocol files"""
    
    def __init__(self, lfw_root, transform=None, pairs_file=None):
        self.lfw_root = lfw_root
        self.transform = transform
        self.pairs = []
        
        if not pairs_file or not os.path.exists(pairs_file):
            raise ValueError(f"Must provide a valid pairs file path. Got: {pairs_file}")
            
        print(f"Loading pairs from: {pairs_file}")
        self._load_pairs_from_file(pairs_file)
        print(f"Loaded {len(self.pairs)} pairs")
    
    def _load_pairs_from_file(self, pairs_file):
        """Load pairs from LFW protocol file"""
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
        
        # First line contains the number of pairs in some files
        first_line = lines[0].strip()
        start_idx = 1 if first_line.isdigit() else 0
        
        # Process remaining lines
        for line_idx in range(start_idx, len(lines)):
            line = lines[line_idx].strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            
            # Format: name idx1 idx2 (same person)
            if len(parts) == 3:
                name, idx1, idx2 = parts
                img1_path = os.path.join(self.lfw_root, name, f"{name}_{idx1.zfill(4)}.jpg")
                img2_path = os.path.join(self.lfw_root, name, f"{name}_{idx2.zfill(4)}.jpg")
                
                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    self.pairs.append((img1_path, img2_path, 1))
                else:
                    print(f"Warning: Could not find images for {name} {idx1} {idx2}")
                    
            # Format: name1 idx1 name2 idx2 (different people)
            elif len(parts) == 4:
                name1, idx1, name2, idx2 = parts
                img1_path = os.path.join(self.lfw_root, name1, f"{name1}_{idx1.zfill(4)}.jpg")
                img2_path = os.path.join(self.lfw_root, name2, f"{name2}_{idx2.zfill(4)}.jpg")
                
                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    self.pairs.append((img1_path, img2_path, 0))
                else:
                    print(f"Warning: Could not find images for {name1} {idx1} and {name2} {idx2}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {img1_path}, {img2_path}")
            print(f"Error: {e}")
            # Return dummy images
            img1 = Image.new('RGB', (224, 224))
            img2 = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return (img1, img2), label

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate on LFW')
    parser.add_argument('--lfw-root', type=str, required=True,
                        help='Path to LFW root directory (containing person folders)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model', type=str, default='spikingresformer_ti',
                        help='Model architecture name')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of time steps')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--input-size', type=int, nargs=2, default=[224,224],
                        help='Model input size (H W)')
    parser.add_argument('--embed-dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--pairs-file', type=str, default=None,
                       help='Path to pairs.txt file (if in a different location)')
    return parser.parse_args()

def extract_embeddings(model, image, T):
    """Extract embeddings from the model, handling spiking neural network specifics"""
    model.eval()
    with torch.no_grad():
        # Forward pass through the model
        output = model(image)  # Shape: [T, B, embed_dim] for spiking models
        
        # Reset the spiking network state
        functional.reset_net(model)
        
        # Average over time steps if output has time dimension
        if len(output.shape) == 3 and output.shape[0] == T:
            embeddings = output.mean(0)  # [B, embed_dim]
        else:
            embeddings = output  # [B, embed_dim]
        
        # L2 normalize embeddings for better similarity computation
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
    return embeddings

def main():
    args = parse_args()
    
    print(f"LFW root directory: {args.lfw_root}")
    print(f"Checking directory structure...")
    
    # Check if the directory exists and has the right structure
    if not os.path.exists(args.lfw_root):
        print(f"Error: LFW root directory does not exist: {args.lfw_root}")
        return
    
    # List some directories to verify structure
    subdirs = [d for d in os.listdir(args.lfw_root) if os.path.isdir(os.path.join(args.lfw_root, d))]
    print(f"Found {len(subdirs)} person directories")
    if len(subdirs) > 0:
        print(f"Sample directories: {subdirs[:5]}")
        
        # Check a sample person directory
        sample_dir = os.path.join(args.lfw_root, subdirs[0])
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
        print(f"Sample person '{subdirs[0]}' has {len(sample_files)} images")
        if len(sample_files) > 0:
            print(f"Sample image names: {sample_files[:3]}")
    
    # 1. Define transforms (same as training)
    val_transforms = transforms.Compose([
        transforms.Resize(tuple(args.input_size)),
        transforms.CenterCrop(tuple(args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    
    # 2. Load LFW pairs dataset with custom implementation
    print("\nLoading LFW pairs dataset...")
    dataset = LFWPairsDataset(
        lfw_root=args.lfw_root, 
        transform=val_transforms,
        pairs_file=args.pairs_file
    )
    
    print(f"Dataset loaded with {len(dataset)} pairs")
    if len(dataset) == 0:
        print("ERROR: No pairs were loaded from the dataset!")
        return
        
    # Check a few sample pairs
    print("\nChecking a few sample pairs:")
    for i in range(min(3, len(dataset))):
        try:
            (img1, img2), label = dataset[i]
            print(f"Sample {i}: img1 shape {img1.shape}, img2 shape {img2.shape}, label {label}")
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
    
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)
    
    print(f"\nDataLoader created with {len(loader)} batches")
    
    # 3. Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with embedding output (not classification)
    try:
        model = create_model(
            args.model,
            T=args.T,
            num_classes=args.embed_dim,  # Use embedding dimension as output
            img_size=args.input_size[0],
        )
        print(f"Model created successfully: {args.model}")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
        print(f"Checkpoint loaded. Keys in checkpoint: {list(ckpt.keys())}")
        
        # Handle different checkpoint structures
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
            
        model.load_state_dict(state_dict)
        print("Model state_dict loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    model.to(device).eval()
    print(f"Model loaded successfully. Evaluating on LFW...")
    
    # 4. Evaluate
    all_labels = []
    all_distances = []
    all_similarities = []
    
    print("\nStarting evaluation loop...")
    with torch.no_grad():
        batch_count = 0
        for batch_idx, ((img1, img2), labels) in enumerate(loader):
            print(f"Processing batch {batch_idx}, size: {img1.shape}")
            img1, img2 = img1.to(device), img2.to(device)
            
            # Extract embeddings for both images
            emb1 = extract_embeddings(model, img1, args.T)
            emb2 = extract_embeddings(model, img2, args.T)
            
            # Compute distances and similarities
            distances = torch.norm(emb1 - emb2, p=2, dim=1).cpu().numpy()
            similarities = torch.sum(emb1 * emb2, dim=1).cpu().numpy()  # cosine similarity
            
            all_distances.append(distances)
            all_similarities.append(similarities)
            all_labels.append(labels.numpy())
            
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Processed {batch_count}/{len(loader)} batches")
    
    print(f"\nProcessed total of {batch_count} batches")
    print(f"all_distances length: {len(all_distances)}")
    print(f"all_similarities length: {len(all_similarities)}")
    print(f"all_labels length: {len(all_labels)}")
    
    if len(all_distances) == 0:
        print("ERROR: No batches were processed during evaluation!")
        return
    
    # Concatenate all results
    all_distances = np.concatenate(all_distances)
    all_similarities = np.concatenate(all_similarities)
    all_labels = np.concatenate(all_labels)
    
    print(f"\nTotal pairs evaluated: {len(all_labels)}")
    print(f"Positive pairs: {np.sum(all_labels)}")
    print(f"Negative pairs: {len(all_labels) - np.sum(all_labels)}")
    
    # 5. Compute metrics using distance (lower distance = more similar)
    fpr_dist, tpr_dist, thresholds_dist = roc_curve(all_labels, -all_distances)
    roc_auc_dist = auc(fpr_dist, tpr_dist)
    
    # 6. Compute metrics using similarity (higher similarity = more similar)
    fpr_sim, tpr_sim, thresholds_sim = roc_curve(all_labels, all_similarities)
    roc_auc_sim = auc(fpr_sim, tpr_sim)
    
    print(f"\n=== Results ===")
    print(f"Distance-based AUC: {roc_auc_dist:.4f}")
    print(f"Similarity-based AUC: {roc_auc_sim:.4f}")
    
    # Find best accuracy with distance threshold
    best_acc_dist = 0
    best_thresh_dist = 0
    for thr in thresholds_dist:
        preds = (all_distances < thr).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc_dist:
            best_acc_dist, best_thresh_dist = acc, thr
    
    # Find best accuracy with similarity threshold
    best_acc_sim = 0
    best_thresh_sim = 0
    for thr in thresholds_sim:
        preds = (all_similarities > thr).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc_sim:
            best_acc_sim, best_thresh_sim = acc, thr
    
    print(f"Best accuracy (distance): {best_acc_dist:.4f} at threshold {best_thresh_dist:.4f}")
    print(f"Best accuracy (similarity): {best_acc_sim:.4f} at threshold {best_thresh_sim:.4f}")
    
    # Summary statistics
    print(f"\n=== Distance Statistics ===")
    print(f"Mean distance (same): {all_distances[all_labels==1].mean():.4f}")
    print(f"Mean distance (different): {all_distances[all_labels==0].mean():.4f}")
    print(f"Std distance (same): {all_distances[all_labels==1].std():.4f}")
    print(f"Std distance (different): {all_distances[all_labels==0].std():.4f}")

if __name__ == '__main__':
    main()