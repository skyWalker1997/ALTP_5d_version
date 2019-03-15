import data_loader
# import  keras.models.load_model
from models import model

model = model()
model.load_weights('model.h5')
print(model.summary())
day_0_c0, day_0_c1, day_0_c2, day_0_c3, day_0_c4, day_1_c0, day_1_c1, day_1_c2, \
    day_1_c3, day_1_c4, day_2_c0, day_2_c1, day_2_c2, day_2_c3, day_2_c4, y = data_loader.test_data()

pred = model.predict(x =[day_0_c0, day_0_c1, day_0_c2, day_0_c3, day_0_c4, day_1_c0,
                      day_1_c1, day_1_c2, day_1_c3, day_1_c4, day_2_c0, day_2_c1, day_2_c2, day_2_c3, day_2_c4],batch_size=3,verbose=2)
for i in range(0,3):
    print(pred[i],y[i])