# Lab5
### Пошаговое затухание (Step-Decay)
 * Синий   
- [x] initial_lrate=0.00001
- [x] drop=0.5
- [x] epoch_drop=5
* Красный 
- [x] initial_lrate=0.00001
- [x] drop=0.8
- [x] epoch_drop=5
* Голубой 
- [x] initial_lrate=0.00001
- [x] drop=0.5
- [x] epoch_drop=10


![step](step1.PNG)
![step](step2.PNG)

Голубой график сходиться быстрее всех ,но имеет большой разброс на валидационной ошибке. Точность и ошибка на валидационной выборке у красного и синего графиков примерно одинакова но красныйсходиться быстрее. 
Оптимальныйми являються параметры соответствующие красному графику. 

### Экспоненциальное затухание (Exponential Decay)
* Оранжевый
- [x] initial_lrate=0.0001
- [x] k=0.1

* Синий
- [x] initial_lrate=0.0001
- [x] k=0.5
* Красный
- [x] initial_lrate=0.00001
- [x] k=0.5

![exp](exp1.PNG)
![exp](exp2.PNG)

### Политика “предварительного разогрева” (Warm-Up)
 1)Пошаговое затухание  
- [x] initial_lrate=0.0001
- [x] drop=0.5
- [x] epoch_drop=5

![exp](warmstep1.PNG)
![exp](stepwarm2.PNG)

2) Экспоненциальное затухание
- [x] initial_lrate=0.0001
- [x] k=0.5
![exp](expwarm1.PNG)
![exp](expwarm2.PNG)


