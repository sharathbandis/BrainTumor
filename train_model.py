import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# 1. Define Image Augmentation (Makes the model robust to different MRI angles)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2 # Reserves 20% of data for testing
)

# NOTE: Update 'data/' to the actual path of your folder containing 'yes' and 'no' subfolders
data_dir = 'data/' 

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224), # MobileNetV2 standard size
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 2. Load Pre-trained MobileNetV2 (Exclude the top categorization layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers so we don't destroy their pre-learned features
for layer in base_model.layers:
    layer.trainable = False

# 3. Add Custom Layers for Brain Tumor Detection
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) # Prevents overfitting
predictions = Dense(1, activation='sigmoid')(x) # Binary output (Tumor / No Tumor)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile and Train
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("Starting training...")
history = model.fit(
    train_generator,
    epochs=10, 
    validation_data=val_generator
)

# 5. Save the upgraded model
model.save('models/powerful_brain_tumor_detector.h5')
print("Model saved successfully!")