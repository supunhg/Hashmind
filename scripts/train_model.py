#!/usr/bin/env python3
"""Train machine learning model for hash identification."""

import json
import pickle
import sys
import os
from collections import Counter
from typing import List, Dict, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.extractor import FeatureExtractor

console = Console()


def load_training_data(filepath: str) -> Tuple[List[str], List[str]]:
    """Load training data from JSONL file."""
    hashes = []
    labels = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading training data...", total=None)
        
        with open(filepath, 'r') as f:
            for line in f:
                sample = json.loads(line)
                hashes.append(sample['hash'])
                labels.append(sample['algorithm'])
        
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Loaded {len(hashes):,} samples")
    return hashes, labels


def extract_features(hashes: List[str]) -> List[Dict]:
    """Extract features from hash strings."""
    extractor = FeatureExtractor()
    features = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Extracting features...", total=len(hashes))
        
        for hash_str in hashes:
            features.append(extractor.extract(hash_str))
            progress.update(task, advance=1)
    
    console.print(f"[green]✓[/green] Extracted features from {len(features):,} samples")
    return features


def train_model(features: List[Dict], labels: List[str], output_path: str):
    """Train XGBoost model."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, accuracy_score
        import numpy as np
        import pandas as pd
    except ImportError:
        console.print("[red]Error: Required ML libraries not installed.[/red]")
        console.print("Install with: pip install xgboost scikit-learn pandas numpy")
        return
    
    console.print("\n[cyan]Preparing data for training...[/cyan]")
    
    df = pd.DataFrame(features)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes
    
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    table = Table(show_header=False, box=box.ROUNDED)
    table.add_row("Training samples", f"{len(X_train):,}")
    table.add_row("Test samples", f"{len(X_test):,}")
    table.add_row("Features", f"{X.shape[1]}")
    table.add_row("Classes", f"{len(label_encoder.classes_)}")
    console.print(table)
    
    console.print("\n[cyan]Training XGBoost model...[/cyan]")
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(label_encoder.classes_),
        max_depth=8,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=10,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    console.print("\n[cyan]Evaluating model...[/cyan]")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    console.print(Panel(
        f"[bold green]Test Accuracy: {accuracy:.2%}[/bold green]",
        box=box.DOUBLE
    ))
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': list(df.columns),
        'accuracy': accuracy,
        'n_features': X.shape[1],
        'n_classes': len(label_encoder.classes_),
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    console.print(f"[green]✓[/green] Model saved to {output_path}")
    
    console.print("\n[cyan]Top 10 Feature Importance:[/cyan]")
    feature_importance = pd.DataFrame({
        'feature': df.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    imp_table = Table(box=box.SIMPLE)
    imp_table.add_column("Feature", style="cyan")
    imp_table.add_column("Importance", justify="right", style="green")
    
    for idx, row in feature_importance.head(10).iterrows():
        imp_table.add_row(row['feature'], f"{row['importance']:.4f}")
    
    console.print(imp_table)


def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train hash identification model')
    parser.add_argument('--data', type=str, default='samples/training_data.jsonl',
                        help='Training data file (JSONL format)')
    parser.add_argument('--output', type=str, default='models/hashmind_model.pkl',
                        help='Output model file')
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]hashmind Model Training[/bold cyan]",
        box=box.DOUBLE
    ))
    
    if not os.path.exists(args.data):
        console.print(f"[red]Error: Training data not found at {args.data}[/red]")
        console.print("Generate training data first:")
        console.print(f"  [cyan]python scripts/generate_training_data.py[/cyan]")
        return 1
    
    hashes, labels = load_training_data(args.data)
    
    label_counts = Counter(labels)
    console.print(f"\n[cyan]Dataset:[/cyan] {len(hashes):,} samples across {len(label_counts)} algorithms")
    
    features = extract_features(hashes)
    train_model(features, labels, args.output)
    
    console.print(Panel.fit(
        "[bold green]✓ Training Complete![/bold green]",
        box=box.DOUBLE
    ))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
