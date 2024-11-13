from matplotlib import pyplot as plt
import numpy as np

countries = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
train_set = [1153, 1449, 711, 1424]
test_set = [136, 140, 100, 136]

x = np.arange(len(countries))
print(x)
width = 0.2
train_set_x = x
test_set_x = x + width
plt.xticks(x, labels=countries)


plt.bar(train_set_x,train_set,width=width,color="blue",label='Train Set')
plt.bar(test_set_x,test_set,width=width,color="green",label='Test Set')
plt.legend()

plt.show()





