import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,Input,Flatten,Dense,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#print(tf.__version__)
#Train data generation

Batch_size = 8

train_datagen = ImageDataGenerator(rescale= 1./255,rotation_range=0.2, shear_range=0.2, zoom_range = 0.2, width_shift_range = 0.2,height_shift_range = 0.2,validation_split=0.2)

train_data = train_datagen.flow_from_directory(r'D:\Python Projects\Drowsiness_Detection\Prepared_data\train',
                                               target_size=(80,80),
                                               batch_size=Batch_size,
                                               class_mode='categorical',
                                               subset='training')

validation_data = train_datagen.flow_from_directory(r'D:\Python Projects\Drowsiness_Detection\Prepared_data\train',
                                                      target_size=(80,80),
                                                        batch_size=Batch_size,
                                                        class_mode='categorical',
                                                        subset='validation')

#Test data generation
test_datagen = ImageDataGenerator(rescale= 1./255)
test_data = test_datagen.flow_from_directory(r'D:\Python Projects\Drowsiness_Detection\Prepared_data\test',
                                               target_size=(80,80),
                                               batch_size=Batch_size,
                                               class_mode='categorical')

bmodel = InceptionV3(include_top=False, weights='imagenet', input_tensor = Input(shape=(80,80,3),batch_size=Batch_size))
hmodel = bmodel.output
hmodel = Flatten()(hmodel)
hmodel = Dense(64, activation='relu')(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(2, activation='softmax')(hmodel)

model = Model(inputs = bmodel.input , outputs = hmodel)

for layer in bmodel.layers:
    layer.trainable = False

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(r'D:\Python Projects\Drowsiness_Detection\Models\model.h5', monitor='val_loss', save_best_only=True,verbose = 3)

earlystop = EarlyStopping(monitor='val_loss', patience=7, verbose=3, restore_best_weights=True)

learning_rate = ReduceLROnPlateau(monitor='val_loss',patience=3,verbose=3)

callbacks = [checkpoint,earlystop,learning_rate]


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_data,
    steps_per_epoch=train_data.samples // Batch_size,
    validation_data=validation_data,
    validation_steps=validation_data.samples // Batch_size,
    callbacks=callbacks,
    epochs=5
)


loss_tr, acc_tr = model.evaluate(train_data)
print(acc_tr, loss_tr)

loss_vr, acc_vr = model.evaluate(validation_data)
print(acc_vr, loss_vr)

loss_test, acc_test = model.evaluate(test_data)
print(acc_test, loss_test)



