!wget -O input_image.jpg https://i.pinimg.com/736x/34/0f/d4/340fd49bc43d4c531fa7da9aea1ee3d6.jpg
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
def bird_detecter(link, model, label):
  # Disable scientific notation for clarity
  np.set_printoptions(suppress=True)

  # Load the model
  model = load_model(model, compile=False)

  # Load the labels
  class_names = open(label, "r").readlines()

  # Create the array of the right shape to feed into the keras model
  # The 'length' or number of images you can put into the array is
  # determined by the first position in the shape tuple, in this case 1
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  # Replace this with the path to your image
  image = Image.open(link).convert("RGB")

  # resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

  # turn the image into a numpy array
  image_array = np.asarray(image)

  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  # Load the image into the array
  data[0] = normalized_image_array

  # Predicts the model
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]

  return(class_name [2:], confidence_score)
name, score = bird_detecter(link = '/content/input_image.jpg', model = '/content/keras_model.h5', label = '/content/labels.txt')
if name.strip() == 'Голуби' and score > 0.7:
  print('ячмень, пшеница, перловка, семечки, гречка, пшено, горох, чечевица и другие сухие крупы.')

elif name.strip() == 'Синички' and score > 0.7:
  print('''
Семечки: подсолнечника, тыквы. Сырые и несолёные, можно в скорлупе.

Орехи: кешью, фундук. Без скорлупы, сырые и несолёные.

Крупы: овёс, просо, гречка. Сырые. Не рекомендуют давать птицам пшено.

Мясо и сало. Маленькие кусочки с большим содержанием жира. Не нужно варить, жарить, солить и приправлять специями.

Ягоды, фрукты и овощи: рябина, клюква, шиповник, калина, вяленые яблоки и сливы, мякоть тыквы и протёртая свежая морковь.''')

elif name.strip() == 'Лебеди' and score > 0.7:
  print('овёс, ячмень, перловка (желательно не твердые, а слегка отваренные), вместе с мелко нарезанными или тёртыми сырыми овощами (капуста, морковь), сухие зерновые корма или птичий комбикорм. Особенно полезными будут пророщенные зерна пшеницы.')

elif name.strip() == 'Курицы' and score > 0.7:
  print('''
пшеница,

овёс (если куры старше 5–6 месяцев),

просо,

ячмень,

кукуруза (если старше 5–6 месяцев, если младше — то молотая).''')

else:
  print('На картине нет птиц')


