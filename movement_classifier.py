from tabpfn import TabPFNClassifier
import numpy as np

# 1. Provide a tiny dataset to teach TabPFN
# Format: [Distance_To_Treasure, Speed_mps, Direction_Changes]
X_train = np.array([
    [500, 0.5, 90],  # Far, slow, turning a lot -> Wandering
    [400, 1.2, 10],  # Far, fast, straight -> Tracking Directly
    [300, 0.3, 180], # Medium distance, slow, reversing -> Wandering
    [50, 2.0, 0]     # Close, fast, straight -> Tracking Directly
])
# Labels: 0 (Wandering), 1 (Tracking)
y_train = np.array([0, 1, 0, 1]) 

# 2. Initialize TabPFN on CPU (Perfect for low-end PCs!)
classifier = TabPFNClassifier(device='cpu')
classifier.fit(X_train, y_train)

def analyze_movement(dist, speed, direction_changes):
    # Predict the user's live movement behavior
    prediction = classifier.predict([[dist, speed, direction_changes]])
    return "Wandering aimlessly" if prediction[0] == 0 else "Tracking directly towards the gold"