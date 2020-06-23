# Lab 4

#### Выбор  максимально возможного темпа обучения

* Оранжевый - lr=0.01
* Синий - lr=0.001
* Красный - lr=0.0001
* Розовый -  lr=0.00001
* Серый -  lr=0.000001
![lr](lr.PNG)

#### Горизонтальное отражение
![lr](flip1.PNG)
![lr](flip2.PNG)
![lr](flip3.PNG)
![lr](flip4.PNG)

#### Поворот на случайный угол
Оранжевый - 15 
синий - 30
Красный - 45
Голубой - 90

![lr](rot1.PNG)
![lr](rot2.PNG)
![lr](rot3.PNG)
![lr](rot4.PNG)

#### Изменение яркости и контраста
*Оранжевый
tf.image.random_brightness(image, 0.6, seed=None)
image = tf.image.random_contrast(image, lower=0.3, upper=1.3, seed=None)    

*синий
tf.image.random_brightness(image, 0.4, seed=None)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.3, seed=None)    

*красный
tf.image.random_brightness(image, 0.4, seed=None)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=None)   
    
![lr](contr1.PNG)
![lr](contr2.PNG)
![lr](contr3.PNG)
![lr](contr4.PNG)


#### Использование случайного участка изображения

Оранжеый - размер участка 100 х 100
Красный - размер участка 150 х 150
Голубой - размер участка 50 х 50 

![lr](crop.PNG)









