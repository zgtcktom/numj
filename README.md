# numj

## Indexing, Slicing and Iterating
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
