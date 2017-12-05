# numpy-pandas-practice

## 簡介
NumPy是Python語言的一個擴充程式庫。支援高階大量的維度陣列與矩陣運算，此外也針對陣列運算提供大量的數學函式函式庫。

## 特色
NumPy參考 **CPython** ，而在這個Python實作直譯器上所寫的數學演算法程式碼，常遠比編譯過的相同程式碼要來得慢。為了解決這個難題，NumPy引入了多維陣列以及可以直接有效率地操作多維陣列的函式與運算子。因此在NumPy上只要能被表示為針對陣列或矩陣運算的演算法，**行效率幾乎都可以與編譯過的等效C語言程式碼一樣快**

---

## [1. numpy 屬性](https://github.com/Airwavess/numpy-pandas-practice/blob/master/1.%20numpy%20attribute.ipynb)
如果要使用 numpy 建立 array 可以使用 `nu.array()`:
```
import numpy as np

array = np.array([[1,2,3],[4,5,6]])
```

查詢 numpy array 的維度可以使用 `[array-name].ndim`:
```
print('number of dimension:', array.ndim)
```

接下來，讓我來看看 numpy array 的形狀長怎樣:
```
print('shape', array.shape)
```

或者是，想知道 numpy array 整體有多少元素，可以使用 `[array-name].size`
```
print('size', array.size)
```
---

## [2. 建立 numpy array](https://github.com/Airwavess/numpy-pandas-practice/blob/master/2.%20Create%20array.ipynb)

一般建立 numpy array 可以使用 `np.array()`，並可以利用 `[array-name].dtpye` 查詢 array 中的元素型態:
```
import numpy as np

array = np.array([2,1,4])

print(array.dtype)
```

如果想要設定 numpy array 的元素型態，可以直接在創建 array 時，設定 `dtype`:
```
array_2 = np.array([3,4,5], dtype=np.float)
one_array = np.ones((3, 5), dtype=np.int)
```

建立 numpy array 有許多方便的方法，例如我們想要建立所有元素皆是 0 的 array 可以使用 `np.zeros()`:
```
zero_array = np.zeros((3, 4))
```

建立所有值皆為 1 的 array 則可以使用 `np.ones()`:
```
one_array = np.ones((3, 5), dtype=np.int)
```

或者是使用 `np.empty()` 建立 array，而不設定裡面的元素，**但是，裡面的元素值會 random 產生**:
```
empty_array = np.empty((2, 1))
# [[ 0.        ]
# [-2.00000012]]
```

`np.arange([start, ]stop, [step, ]dtype=None)` 是一個非常方便的函式，可以依照使用者設定的參數填入值:
```
print(np.arange(10, 18, 2))    
# [10 12 14 16]

print(np.arange(12).reshape(3,4))
# [[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]
```

最後一個要介紹的是 `numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)`，這個函式與 `np.arrange()`很像，皆是依照給定的參數回傳區間內的陣列：
```
print(np.linspace(1, 10, 20))
#[  1.           1.47368421   1.94736842   2.42105263   2.89473684
#   3.36842105   3.84210526   4.31578947   4.78947368   5.26315789
#   5.73684211   6.21052632   6.68421053   7.15789474   7.63157895
#   8.10526316   8.57894737   9.05263158   9.52631579  10.        ]

print(np.linspace(1, 10, 6).reshape(2, 3))
# [[  1.    2.8   4.6]
#  [  6.4   8.2  10. ]]
```
