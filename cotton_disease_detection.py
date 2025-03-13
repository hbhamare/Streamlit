import os
import zipfile
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class CottonDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CottonDiseaseClassifier, self).__init__()
        # Load pretrained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x

def get_severity(confidence):
    if confidence < 0.8:
        return "Mild"
    elif 0.8 < confidence < 0.9:
        return "Moderate"
    elif confidence > 0.9:
        return "High"

def download_dataset():
    """Download the dataset from Kaggle"""
    if not os.path.exists('train'):
        print("Downloading dataset from Kaggle...")
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'meftahuljannat/cotton-leaf-diseases-merged',
                path='.',
                unzip=True
            )
            print("Dataset downloaded and extracted successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nPlease ensure you have:")
            print("1. Installed kaggle package: pip install kaggle")
            print("2. Downloaded kaggle.json from your Kaggle account")
            print("3. Placed kaggle.json in ~/.kaggle/ directory")
            print("4. Set correct permissions: chmod 600 ~/.kaggle/kaggle.json")
            return False
    else:
        print("Dataset directory 'train' already exists!")
    
    # Verify the dataset structure
    expected_classes = ['Aphids', 'Army worm', 'Bacterial Blight', 'Curl Virus',
                       'Fussarium Wilt', 'Healthy', 'Powdery Mildew', 'Target spot']
    
    all_classes_exist = all(os.path.exists(os.path.join('train', class_name)) 
                           for class_name in expected_classes)
    
    if not all_classes_exist:
        print("Warning: Dataset directory structure is not as expected!")
        print("Expected classes:", expected_classes)
        print("Found directories:", os.listdir('train'))
        return False
        
    return True

class CottonDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))  # Sort to ensure consistent class ordering
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_data_loaders(data_dir, batch_size=32):
    """Create train, validation, and test data loaders"""
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    full_dataset = CottonDiseaseDataset(data_dir, transform=train_transform)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Override transforms for validation and test sets
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, full_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='Training')
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({'loss': running_loss/len(train_loader), 
                                'accuracy': 100.*correct/total})

    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss/len(val_loader), 100.*correct/total

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def main():
    # Download and verify dataset
    if not download_dataset():
        print("Error with dataset. Please fix the issues above and try again.")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            data_dir='train',
            batch_size=32
        )
        print(f"Found {len(class_names)} classes: {class_names}")
        
        # Initialize model
        print("Initializing model...")
        model = CottonDiseaseClassifier(num_classes=len(class_names))
        model = model.to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.1
        )
        
        # Training loop
        print("\nStarting training...")
        num_epochs = 20
        best_val_acc = 0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'class_names': class_names,
                    'val_acc': val_acc
                }, 'best_model.pth')
            
            # Record metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
        
        # Plot training history
        print("\nPlotting training history...")
        plot_training_history(train_losses, train_accs, val_losses, val_accs)
        
        # Load best model and evaluate
        print("\nEvaluating best model...")
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate_model(model, test_loader, device, class_names)
        
        print("\nTraining completed successfully!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print("Model saved as 'best_model.pth'")
        print("Training plots saved as 'training_history.png'")
        print("Confusion matrix saved as 'confusion_matrix.png'")
        
    except Exception as e:
        print(f"\nAn error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()

def predict_image(model, image_path, class_names, device):
    """Make prediction on a single image"""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    return class_names[predicted], confidence.item()

def test_model():
    """Test the trained model with a sample image"""
    # Load the trained model
    checkpoint = torch.load('best_model.pth')
    model = CottonDiseaseClassifier(num_classes=len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get class names
    class_names = checkpoint['class_names']
    
    # Test with a sample image from the test set
    test_dir = 'test'
    random_class = np.random.choice(os.listdir(test_dir))
    random_image = np.random.choice(os.listdir(os.path.join(test_dir, random_class)))
    image_path = os.path.join(test_dir, random_class, random_image)
    
    # Make prediction
    predicted_class, confidence = predict_image(model, image_path, class_names, device)
    
    # Display results with severity
    severity = get_severity(confidence)
    print(f"\nTesting model with image: {image_path}")
    print(f"Actual class: {random_class}")
    print(f"Predicted class: {predicted_class}")
    # print(f"Confidence: {confidence:.2%}")
    print(f"Severity: {severity}")
    
    # Show the image
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f"Actual: {random_class}\nPredicted: {predicted_class} ({confidence:.2%})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # main()
    test_model()
