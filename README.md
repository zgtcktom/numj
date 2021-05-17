# numj

## Indexing, Slicing and Iterating
Code:
```
print(">>>", "a = np.arange(10)**3");
NDArray<Double> a = arange(10).astype(NDArray::Double);
a = power(a, array(3).astype(NDArray::Double));

print(">>>", "a");
print(a);

print(">>>", "a[2]");
print(a.item(new int[]{2}));

print(">>>", "a[2:5]");
print(a.get(slice(2, 5)));

print(">>>", "a[:6:2] = 1000");
a.get(slice(null, 6, 2)).set(1000.0);

print(">>>", "a");
print(a);

print(">>>", "a[::-1]");
print(a.get(slice(null, null, -1)));

print(">>>", "for i in a:");
print("...", "\tprint(i**(1/3.))");
for (int[] index : ndindex(a.shape)) {
    print(Math.pow(a.item(index), 1 / 3.));
}
```
Outputs:
```
>>> a = np.arange(10)**3
>>> a
[0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0, 729.0]
>>> a[2]
8.0
>>> a[2:5]
[8.0, 27.0, 64.0]
>>> a[:6:2] = 1000
>>> a
[1000.0, 1.0, 1000.0, 27.0, 1000.0, 125.0, 216.0, 343.0, 512.0, 729.0]
>>> a[::-1]
[729.0, 512.0, 343.0, 216.0, 125.0, 1000.0, 27.0, 1000.0, 1.0, 1000.0]
>>> for i in a:
... 	print(i**(1/3.))
9.999999999999998
1.0
9.999999999999998
3.0
9.999999999999998
4.999999999999999
5.999999999999999
6.999999999999999
7.999999999999999
8.999999999999998
```
