# MULTICLASS 99%+ OPTIMIZATION: Combined Advanced Techniques
# This cell implements: Extended Training + Attention + Cosine Annealing + Feature Engineering + SMOTE

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout, BatchNormalization, Reshape, Input, Layer, Multiply, Permute, Lambda
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import math

print("\n" + "=" * 100)
print(" " * 30 + "MULTICLASS 99%+ OPTIMIZATION")
print("=" * 100)

# ==================== STEP 1: FEATURE ENGINEERING ====================
print("\n[1/6] Feature Engineering: Creating interaction features...")

def create_interaction_features(X, n_top_interactions=20):
    """Create polynomial and interaction features from top correlated features"""
    # Simple squared features for top features (by variance)
    feature_vars = np.var(X, axis=0)
    top_indices = np.argsort(feature_vars)[-n_top_interactions:]
    
    interactions = []
    for i in range(min(10, len(top_indices))):
        idx = top_indices[i]
        interactions.append((X[:, idx] ** 2).reshape(-1, 1))  # Squared
        
    if len(interactions) > 0:
        X_enhanced = np.hstack([X] + interactions)
        print(f"   ✓ Added {len(interactions)} interaction features: {X.shape[1]} → {X_enhanced.shape[1]}")
        return X_enhanced
    return X

x_train_enhanced = create_interaction_features(x_train_full)
x_val_enhanced = create_interaction_features(x_val_full)
x_test_enhanced = create_interaction_features(x_test_full)
n_features_enhanced = x_train_enhanced.shape[1]

# ==================== STEP 2: SMOTE DATA AUGMENTATION ====================
print("\n[2/6] SMOTE Augmentation: Balancing minority classes...")

train_labels_flat = np.argmax(y_train_full, axis=1)
class_counts = np.bincount(train_labels_flat)
print(f"   Original class distribution: {class_counts}")

# Apply SMOTE with k_neighbors=3 (safer for rare classes)
smote = SMOTE(sampling_strategy='not majority', k_neighbors=3, random_state=42)
x_train_smote, y_train_smote_labels = smote.fit_resample(x_train_enhanced, train_labels_flat)

# Convert back to one-hot
y_train_smote = tf.keras.utils.to_categorical(y_train_smote_labels, num_classes=10)
print(f"   ✓ SMOTE applied: {len(x_train_enhanced):,} → {len(x_train_smote):,} samples")
print(f"   New class distribution: {np.bincount(y_train_smote_labels)}")

# ==================== STEP 3: ATTENTION MECHANISM ====================
print("\n[3/6] Building Attention-Enhanced BiGRU Architecture...")

class AttentionLayer(Layer):
    """Attention mechanism for sequence outputs"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                shape=(input_shape[-1], input_shape[-1]),
                                initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                shape=(input_shape[-1],),
                                initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

def build_attention_model(n_features):
    """Build advanced BiGRU model with attention mechanism"""
    inputs = Input(shape=(n_features,))
    
    # Reshape for BiGRU
    x = Reshape((n_features, 1))(inputs)
    x = Reshape((1, n_features))(x)
    
    # BiGRU layers with attention
    x1 = Bidirectional(GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x1 = BatchNormalization()(x1)
    
    x2 = Bidirectional(GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x1)
    x2 = BatchNormalization()(x2)
    
    x3 = Bidirectional(GRU(96, return_sequences=True, dropout=0.2))(x2)
    x3 = BatchNormalization()(x3)
    
    # Attention mechanism
    attention_output = AttentionLayer()(x3)
    
    # Dense classification head
    x = Dense(256, activation='relu')(attention_output)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='BiGRU_Attention_SOTA')
    return model

model_attention = build_attention_model(n_features_enhanced)
print(f"   ✓ Model built with {model_attention.count_params():,} parameters")

# ==================== STEP 4: COSINE ANNEALING WITH WARMUP ====================
print("\n[4/6] Setting up Cosine Annealing LR Schedule with Warmup...")

def cosine_annealing_with_warmup(epoch, lr, warmup_epochs=3, total_epochs=30, max_lr=1e-3, min_lr=1e-6):
    """Cosine annealing learning rate with warmup"""
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

lr_scheduler = LearningRateScheduler(lambda epoch: cosine_annealing_with_warmup(epoch, 0, total_epochs=30))
print("   ✓ Cosine annealing: 3 warmup epochs, max_lr=1e-3, min_lr=1e-6")

# ==================== STEP 5: COMPILE WITH LABEL SMOOTHING ====================
print("\n[5/6] Compiling model with label smoothing (0.1)...")

# Label smoothing: smooth hard labels to prevent overconfidence
label_smoothing = 0.1
model_attention.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
             tf.keras.metrics.Recall(name='recall')]
)
print("   ✓ Label smoothing reduces overconfidence and improves generalization")

# ==================== STEP 6: EXTENDED TRAINING ====================
print("\n[6/6] Training for 30 epochs with callbacks...")

callbacks = [
    lr_scheduler,
    EarlyStopping(monitor='val_accuracy', mode='max', patience=8, 
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('/home/ashish/paper1_t1/gcn_bigru_attention_best.h5',
                   monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
]

print("\n" + "-" * 100)
print("TRAINING CONFIGURATION")
print("-" * 100)
print(f"  Training samples : {len(x_train_smote):,} (SMOTE augmented)")
print(f"  Validation samples : {len(x_val_enhanced):,}")
print(f"  Features : {n_features_enhanced} (with interactions)")
print(f"  Epochs : 30 (early stopping enabled)")
print(f"  Batch size : 256")
print(f"  Optimizer : Adam with cosine annealing")
print(f"  Label smoothing : {label_smoothing}")
print("-" * 100)

import time
start_time = time.time()

history_advanced = model_attention.fit(
    x_train_smote.astype(np.float32), 
    y_train_smote.astype(np.float32),
    validation_data=(x_val_enhanced.astype(np.float32), y_val_full.astype(np.float32)),
    epochs=30,
    batch_size=256,
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)

elapsed = time.time() - start_time
print(f"\n✓ Training completed in {elapsed/60:.1f} minutes")

# ==================== EVALUATION ====================
print("\n" + "=" * 100)
print("EVALUATION: ATTENTION + SMOTE + COSINE ANNEALING + FEATURE ENGINEERING")
print("=" * 100)

# Load best weights
model_attention.load_weights('/home/ashish/paper1_t1/gcn_bigru_attention_best.h5')

# Evaluate
test_results = model_attention.evaluate(x_test_enhanced.astype(np.float32), 
                                       y_test_full.astype(np.float32), 
                                       batch_size=256, verbose=0)
test_loss, test_acc, test_prec, test_rec = test_results

y_pred_advanced = model_attention.predict(x_test_enhanced.astype(np.float32), batch_size=256, verbose=0)
y_pred_advanced_labels = np.argmax(y_pred_advanced, axis=1)
y_test_labels = np.argmax(y_test_full, axis=1)

from sklearn.metrics import f1_score, classification_report
f1_advanced = f1_score(y_test_labels, y_pred_advanced_labels, average='weighted', zero_division=0)

print(f"\n{'Metric':<25} {'Value':<15}")
print("-" * 40)
print(f"{'Test Accuracy':<25} {test_acc*100:.4f}%")
print(f"{'Test Loss':<25} {test_loss:.6f}")
print(f"{'Weighted Precision':<25} {test_prec:.6f}")
print(f"{'Weighted Recall':<25} {test_rec:.6f}")
print(f"{'Weighted F1-Score':<25} {f1_advanced:.6f}")

baseline_acc = 0.9776  # Previous baseline
print(f"\n{'Baseline Accuracy':<25} {baseline_acc*100:.4f}%")
print(f"{'Improvement':<25} {(test_acc-baseline_acc)*100:+.4f}%")

if test_acc >= 0.99:
    print("\n" + "🎉" * 40)
    print("✓✓✓ TARGET ACHIEVED: 99%+ MULTICLASS ACCURACY!")
    print("🎉" * 40)
elif test_acc >= 0.985:
    print("\n✓✓ Excellent! 98.5%+ achieved - approaching 99%")
    print("   Further gains possible with ensemble voting")
else:
    print(f"\n✓ Improved to {test_acc*100:.2f}% (was 97.76%)")
    print("   Next step: Ensemble 3-5 models for additional 0.5-1.0% boost")

print("\n" + "=" * 100)
print("Per-Class Performance:")
print("=" * 100)
print(classification_report(y_test_labels, y_pred_advanced_labels, zero_division=0))

model_attention.save('/home/ashish/paper1_t1/gcn_bigru_multiclass_advanced.h5')
print("\n✓ Model saved: gcn_bigru_multiclass_advanced.h5")
