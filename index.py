import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

# Set dataset paths
train_dir = 'Dataset/training/'
val_dir = 'Dataset/val/'
test_dir = 'Dataset/testing/'

# Image params
img_size = (128, 128)
batch_size = 32

# Load datasets
train_ds = image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size, label_mode='binary', shuffle=True)
val_ds = image_dataset_from_directory(
    val_dir, image_size=img_size, batch_size=batch_size, label_mode='binary', shuffle=False)
test_ds = image_dataset_from_directory(
    test_dir, image_size=img_size, batch_size=batch_size, label_mode='binary', shuffle=False)

# Prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Compute class weights
train_labels = np.array([label.numpy().item() for _, label in train_ds.unbatch()])
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Load base model (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze base model

# Build model
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[reduce_lr, early_stop],
    class_weight=class_weights
)

# Plot accuracy & loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss")
plt.show()

# Evaluate
test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.2f}")
print(f"Test Precision: {test_prec:.2f}")
print(f"Test Recall: {test_rec:.2f}")
print(f"Test AUC: {test_auc:.2f}")

# Predict on test set
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype(int).flatten())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

# Show misclassified examples
for images, labels in test_ds.take(1):
    preds = model.predict(images)
    for i in range(len(images)):
        pred = preds[i][0]
        true = labels[i].numpy()
        if (pred > 0.5 and true == 0) or (pred <= 0.5 and true == 1):
            plt.imshow(images[i].numpy().astype("uint8"))
            pred_label = "Real" if pred > 0.5 else "Fake"
            true_label = "Real" if true == 1 else "Fake"
            plt.title(f"Pred: {pred_label} ({pred:.2f})\nTrue: {true_label}")
            plt.axis("off")
            plt.show()

# Save model
model.save("face_classifier_transfer.keras")
