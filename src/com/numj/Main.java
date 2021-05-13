package com.numj;


import java.util.Arrays;

import static com.numj.NDArray.*;

public class Main {

    static void test() {

        print("NDArray");

        NDArray<Double> ndarray = new NDArray<>(new int[]{1, 10, 21, 2});
        double n = 0;
        for (int i = 0; i < ndarray.shape[0]; i++) {
            for (int j = 0; j < ndarray.shape[1]; j++) {
                for (int k = 0; k < ndarray.shape[2]; k++) {
                    for (int l = 0; l < ndarray.shape[3]; l++) {
                        ndarray.data(
                                (i * ndarray.shape[1] * ndarray.shape[2] * ndarray.shape[3])
                                        + (j * ndarray.shape[2] * ndarray.shape[3])
                                        + (k * ndarray.shape[3])
                                        + l, n);
                        n++;
                    }
                }
            }
        }
        print(String.valueOf(ndarray.data.data.length));
        print(Arrays.toString(ndarray.shape));
        print(Arrays.toString(ndarray.data.data));

        NDArray<Double> _view = ndarray
                .get(new Selection[]{
                        index(0), slice(1, 7), slice(), index(0)})
                .get(new Selection[]{index(-2)})
                .get(slice(20, null, -1));

        print("[NDArray]. getValue(): " +
                Arrays.toString
                        (
                                _view.shape
                        ));

        print("[NDArray]. getValue(): " +
                (
                        _view
                                .toString()
                ));

        print(ndarray.toString());


        NDArray<Double> x1 = array(new Double[][]{
                {0.0, 1.0, 2.0},
                {3.0, 4.0, 5.0},
                {6.0, 7.0, 8.0}
        });

        NDArray<Double> x2 = array(new Double[][]{
                {0.0, 1.0, 2.0},
                {3.0, -1.0, 5.0},
                {6.0, 7.0, 8.0}
        });

        print(Arrays.toString(x1.data.data));
        print("x1: " + x1);
        print("x2: " + x2);

        NDArray<Double> out = add(x1, x2);

        print(out.toString());

        NDArray<Double> x = array(new Double[]{1.0, 2.0, 3.0});
        NDArray<Double> y = array(new Double[][]{{4.0}, {5.0}, {6.0}});
        print("x: " + x);
        print("y: " + y);

        print("x + y: " + add(x, y));
        print("x + 2: " + add(x, array(new Double[]{2.0})));
        print("2 + y: " + add(array(new Double[]{2.0}), y));

        NDArray<Double> view = _view.view();
        NDArray<Double> copy = _view.copy();
        print("original: " + _view);
        print("view: " + view);
        print("copy: " + copy);

        view.setValue(new int[]{0}, -999.0);
        copy.setValue(new int[]{1}, 999.0);

        print("original: " + _view);
        print("view: " + view);
        print("copy: " + copy);

        print(add(mul(x, y), div(sub(y, x), x)).toString());

        NDArray<Integer> a = arange(0, 6, 1).reshape(new int[]{3, 2});
        print(a);
        a = array(new Integer[][]{{1, 2, 3}, {4, 5, 6}});
        a = a.reshape(new int[]{3, -1});
        a = a.get(slice(null, null, -1), slice(null, null, -1));
        print(a);

        NDArray<Double> d = arange(0, 21*2, 1).reshape(new int[]{21, 2}).astype(Double::valueOf);
        d = amin(d, 1);
        print(d);

        NDArray<Double> arr = NDArray.random(new int[]{21, 2});
        NDArray<Double> arr1 = NDArray.sub(arr, NDArray.amin(arr, 0));
        print("arr:", arr);
        print("arr-amin(arr):", arr1);
        print("arr>0.5", NDArray.gt(arr1, array(new Double[]{0.5})));

        print(argmax(NDArray.arange(0, 100, 1).astype(Double::valueOf)));

        arr.get(1).set(array(new Double[]{99.0, 99.1}));
        print(arr);

        NDArray<Double> concat0 = NDArray.random(new int[]{1, 2});
        NDArray<Double> concat1 = NDArray.random(new int[]{2, 2});
        print(concat0);
        print(concat1);
        print(concat(concat0, concat1));
        print(Arrays.toString(concat0.get(slice(1, null)).shape));

        print(new NDArray<Double>(new int[]{0, 2, 9}));

        print(mean(arange(0,100,1).reshape(new int[]{5, 5, -1}).astype(NDArray::Double), 1));

        print(abs(sub(random(new int[]{5,2}), array(new Double[]{0.5}))));

        for (int[] index:NDArray.ndindex(new int[]{1,5,2})) {
            print(Arrays.toString(index));
        }
        print(NDArray.ndindex(new int[]{1,5,2}));

        print("END");
    }

    public static void main(String[] args) {
        test();
    }
}
