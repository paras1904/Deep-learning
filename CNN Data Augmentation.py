from keras.preprocessing.image import ImageDataGenerator , img_to_array, array_to_img,load_img

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
img = load_img('/home/paras/Data/1.jpg')
x = img_to_array(img)
x = x.reshape((1,)+x.shape)
i = 0
for batch in datagen.flow(x,batch_size=1,save_to_dir='preview',
                          save_prefix='cat',save_format='jpg'):
    i+=1
    if i >20:
        break