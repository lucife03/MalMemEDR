import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, n_features=None):
        self.scaler = StandardScaler()
        self.selector = VarianceThreshold(threshold=0.01)
        self.n_features = n_features
        self.selected_features = None
        self.feature_names = None
        print(f"Initializing preprocessor {'with all features' if n_features is None else f'with {n_features} features'}")
    
    def load_data(self, file_path):
        try:
            # Load data
            df = pd.read_excel(file_path)
            print("Initial shape:", df.shape)
            print("Columns:", df.columns.tolist())  # Debug print
            
            # Get the first column (it might be unnamed or have a different name)
            first_col = df.columns[0]  # Get the name of the first column
            print(f"First column name: {first_col}")  # Debug print
            
            # Store malware type information from the first column
            malware_types = df[first_col].copy()
            
            # Drop problematic columns
            columns_to_drop = [first_col, 'pslist.nprocs64bit', 'handles.nport', 
                              'svcscan.interactive_process_services']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            print("\nProcessing features...")
            # Handle string columns
            for column in df.columns:
                if df[column].dtype == 'object':
                    if column == "Class":
                        continue
                    else:
                        try:
                            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)
                        except:
                            print(f"Dropping column {column} due to conversion error")
                            df.drop(columns=[column], inplace=True)
            
            # Convert class labels
            df['Class_Malware'] = df['Class'].apply(lambda x: 1 if 'Malware' in str(x) else 0)
            df.drop(columns=['Class'], inplace=True)
            
            # Store feature names
            self.feature_names = [col for col in df.columns if col != 'Class_Malware']
            
            print("\nFinal shape:", df.shape)
            print("Features remaining:", len(df.columns) - 1)
            return df
            
        except Exception as e:
            print(f"Error in load_data: {str(e)}")
            print("Full column list:", df.columns.tolist())  # Debug print
            raise

    def extract_features(self, X, y=None, train=True):
        try:
            if train:
                print("\nFeature extraction process:")
                print(f"Initial features: {X.shape[1]}")
                
                # 1. Store original feature names
                original_features = X.columns.tolist()
                
                # 2. Remove low variance features
                self.selector = VarianceThreshold(threshold=0.001)
                X_var = self.selector.fit_transform(X)
                
                # Get indices of selected features after variance threshold
                selected_mask = self.selector.get_support()
                selected_features = [f for f, selected in zip(original_features, selected_mask) if selected]
                
                print(f"Features after variance threshold: {X_var.shape[1]}")
                
                # 3. Feature importance with XGBoost
                xgb_model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                xgb_model.fit(X_var, y)
                
                # 4. Create feature importance DataFrame with correct features
                feature_importances = pd.DataFrame({
                    'feature': selected_features,
                    'importance': xgb_model.feature_importances_
                })
                feature_importances = feature_importances.sort_values('importance', ascending=False)
                
                # 5. Select top 24 features or all if less than 24
                n_features = min(24, len(feature_importances))
                top_features = feature_importances.head(n_features)
                
                # Store selected feature indices
                self.selected_features = [selected_features.index(feat) for feat in top_features['feature']]
                self.feature_names = top_features['feature'].tolist()
                
                # Print selected features and their importance
                print("\nSelected features and their importance scores:")
                for idx, row in top_features.iterrows():
                    print(f"{row['feature']}: {row['importance']:.4f}")
                
                # 6. Group features by category
                feature_categories = {
                    'Process': ['pslist', 'dlllist'],
                    'Handles': ['handles'],
                    'Memory': ['malfind'],
                    'System': ['ldrmodules', 'svcscan'],
                    'Callbacks': ['callbacks']
                }
                
                category_counts = {cat: 0 for cat in feature_categories}
                for feature in self.feature_names:
                    for category, prefixes in feature_categories.items():
                        if any(prefix in feature for prefix in prefixes):
                            category_counts[category] += 1
                
                print("\nFeature distribution across categories:")
                for category, count in category_counts.items():
                    print(f"{category}: {count} features")
                
                # 7. Final feature selection and scaling
                X_selected = X_var[:, self.selected_features]
                X_final = self.scaler.fit_transform(X_selected)
                
                print(f"\nFinal feature shape: {X_final.shape}")
                return X_final
                
            else:
                # For inference
                X_var = self.selector.transform(X)
                X_selected = X_var[:, self.selected_features]
                return self.scaler.transform(X_selected)
                
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            print("Debug info:")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape if y is not None else 'None'}")
            print(f"Number of original features: {len(original_features)}")
            print(f"Number of selected features after variance: {len(selected_features)}")
            raise

class MalwareDetector(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(MalwareDetector, self).__init__()
        
        self.input_size = input_size  # Store input size
        self.hidden_size = hidden_size  # Store hidden size
        
        self.feature_extractor = nn.Sequential(
            # First layer - Feature extraction
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            # Second layer - Feature combination
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2),
            nn.Dropout(0.3),
        )
        
        self.classifier = nn.Sequential(
            # Third layer - Classification
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//4),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(hidden_size//4, 2)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

class MalwareDetectorPipeline:
    def __init__(self, input_size, hidden_size=512, learning_rate=0.0002):  # Adjusted learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = MalwareDetector(input_size, hidden_size).to(self.device)
        
        # Use weighted loss for imbalanced classes
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(self.device))
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        self.history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):  # Increased epochs
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Training
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.LongTensor(y_train).to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y_train_tensor.size(0)
            correct += (predicted == y_train_tensor).sum().item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.LongTensor(y_val).to(self.device)
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_acc = (val_predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
                
                # Store metrics
                train_acc = correct / total
                self.history['train_losses'].append(loss.item())
                self.history['train_accuracies'].append(train_acc)
                self.history['val_losses'].append(val_loss.item())
                self.history['val_accuracies'].append(val_acc)
                
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}')
                print(f'Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}')
                
                # Early stopping with higher patience
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_val_acc': best_val_acc
                    }, 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= 7:  # Increased patience
                    print("Early stopping triggered")
                    break
                
                self.scheduler.step(val_acc)

    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_losses'], label='Train Loss')
        plt.plot(self.history['val_losses'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_accuracies'], label='Train Acc')
        plt.plot(self.history['val_accuracies'], label='Val Acc')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def save_model(self, path, preprocessor=None):
        """Save the trained model and preprocessor state"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size
        }
        
        if preprocessor:
            save_dict['preprocessor_state'] = {
                'scaler': preprocessor.scaler,
                'selector': preprocessor.selector,
                'selected_features': preprocessor.selected_features,
                'feature_names': preprocessor.feature_names
            }
        
        torch.save(save_dict, path)
        print(f"Model and preprocessor saved to {path}")

    # Function to load the model for inference
    def load_model(self, path):
        """Load the trained model and preprocessor state"""
        checkpoint = torch.load(path)
        
        # Recreate the model
        model = MalwareDetector(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load preprocessor if available
        preprocessor = None
        if 'preprocessor_state' in checkpoint:
            preprocessor = DataPreprocessor()
            preprocessor.scaler = checkpoint['preprocessor_state']['scaler']
            preprocessor.selector = checkpoint['preprocessor_state']['selector']
            preprocessor.selected_features = checkpoint['preprocessor_state']['selected_features']
            preprocessor.feature_names = checkpoint['preprocessor_state']['feature_names']
        
        return model, preprocessor

if __name__ == "__main__":
    try:
        print("\n=== Starting Malware Detection Pipeline ===")
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Load and preprocess data
        print("Loading Data...")
        preprocessor = DataPreprocessor(n_features=None)
        df = preprocessor.load_data('data (1).xlsx')
        
        print("\nDataset Statistics:")
        print(f"Total samples: {len(df):,}")
        print(f"Malware samples: {len(df[df['Class_Malware'] == 1]):,}")
        print(f"Benign samples: {len(df[df['Class_Malware'] == 0]):,}")
        
        # Prepare data
        y = df["Class_Malware"].values
        X = df.drop(columns=["Class_Malware"])
        
        # Validate shapes
        print(f"\nInitial feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Process features
        X_processed = preprocessor.extract_features(X, y, train=True)
        print(f"Processed feature shape: {X_processed.shape}")
        
        # Split data (90-10 split)
        print("\nSplitting Data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, 
            test_size=0.1, 
            random_state=42, 
            stratify=y
        )
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        
        # Initialize model
        print("\nInitializing Model...")
        input_size = X_processed.shape[1]
        pipeline = MalwareDetectorPipeline(
            input_size=input_size,
            hidden_size=512,
            learning_rate=0.0002
        )
        
        # Train model
        print("\nStarting Training...")
        pipeline.train(X_train, y_train, X_val, y_val, epochs=50)
        
        # Plot training history
        pipeline.plot_training_history()
        
        # Final evaluation
        print("\nFinal Evaluation...")
        pipeline.model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(pipeline.device)
            outputs = pipeline.model(X_val_tensor)
            _, predictions = torch.max(outputs.data, 1)
            val_preds = predictions.cpu().numpy()
        
        # Print final metrics
        print("\nFinal Model Performance:")
        print("-" * 50)
        print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
        print(f"Precision: {precision_score(y_val, val_preds):.4f}")
        print(f"Recall: {recall_score(y_val, val_preds):.4f}")
        print(f"F1 Score: {f1_score(y_val, val_preds):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y_val, val_preds):.4f}")
        
        # Save model with preprocessor state
        print("\nSaving model and preprocessor...")
        model_save_path = 'malware_detector_abs_final.pth'
        torch.save({
            'model_state_dict': pipeline.model.state_dict(),
            'optimizer_state_dict': pipeline.optimizer.state_dict(),
            'scheduler_state_dict': pipeline.scheduler.state_dict(),
            'history': pipeline.history,
            'input_size': pipeline.model.input_size,
            'hidden_size': pipeline.model.hidden_size,
            'preprocessor_state': {
                'scaler': preprocessor.scaler,
                'selector': preprocessor.selector,
                'selected_features': preprocessor.selected_features,
                'feature_names': preprocessor.feature_names
            }
        }, model_save_path)
        print(f"Model and preprocessor saved to {model_save_path}")
        
        print("\n=== Pipeline Complete! ===")
        
    except Exception as e:
        print("\nError in execution:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        import traceback
        traceback.print_exc()