# Lab 3

Обучение нейронных сетей с использованием техники Transfer Learning

# 2b 
Cеть с lr=0.0001 на обучающей выборке сходится быстрее всех.
На обучающей выборке  эксперименты с темпом обучения 0.001 и 0.0001 ведут  себя практически одинаково. Разброс по точности у всех на валидационной выборке практически одинаковый.


* lr=0.01 - оранжевый
* lr=0.001 - синий
* lr=0.0001 - красный


![acc](2c11.PNG)
![loss](2c2.PNG)
![val_cat](2c3.PNG)
![val_loss](2c4.PNG)

Оптимальный вариант - lr=0.0001

# 3с
* lr=0.01 - оранжевый
* lr=0.001 - синий
* lr=0.0001 - красный

![acc](3c.PNG)


# 3d
* lr=0.01 - Cиний
* lr=0.001 - Красный
* lr=0.0001 - Голубой

![acc](3d1.PNG)
![loss](3d2.PNG)
![val_cat](3d3.PNG)
![val_loss](3d4.PNG)

Голубой график сходится быстрее всех и обладает наименьшей ошибкой на валидационной выборке. В связи с этим оптимальный  lr=0.0001
# 3e
* lr=0.01 - оранжевый
* lr=0.001 - синий
* lr=0.0001 - красный

![acc](3e.PNG)
![loss](3e2.PNG)
![val_cat](3e3.PNG)
![val_loss](3e4.PNG)

Синий и красный графики ведут себя практически одинаково.
