package com.numj;


import java.lang.Boolean;
import java.lang.Double;
import java.lang.Integer;
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
                        ndarray.itemset(
                                (i * ndarray.shape[1] * ndarray.shape[2] * ndarray.shape[3])
                                        + (j * ndarray.shape[2] * ndarray.shape[3])
                                        + (k * ndarray.shape[3])
                                        + l, n);
                        n++;
                    }
                }
            }
        }
        print(String.valueOf(ndarray.size));
        print(Arrays.toString(ndarray.shape));

        NDArray<Double> _view = ndarray
                .get(new Selection[]{
                        index(0), slice(1, 7), slice(), index(0)})
                .get(new Selection[]{index(-2)})
                .get(slice(20, null, -1));

        print("[NDArray]. item(): " +
                Arrays.toString
                        (
                                _view.shape
                        ));

        print("[NDArray]. item(): " +
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

        view.set(new int[]{0}, -999.0);
        copy.set(new int[]{1}, 999.0);

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

        NDArray<Double> d = arange(0, 21 * 2, 1).reshape(new int[]{21, 2}).astype(Double::valueOf);
        d = amin(d, 1);
        print(d);

        NDArray<Double> arr = NDArray.random(new int[]{21, 2});
        NDArray<Double> arr1 = NDArray.sub(arr, NDArray.amin(arr, 0));
        print("arr:", arr);
        print("arr-amin(arr):", arr1);
        print("arr>0.5", NDArray.gt(arr1, array(new Double[]{0.5})));

        print(argmax(NDArray.arange(0, 100, 1).astype(Double::valueOf)));

        arr.get(index(1)).set(array(new Double[]{99.0, 99.1}));
        print(arr);
        arr.set(array(new Double[]{99.0, 99.1}));
        print(arr);

        NDArray<Double> concat0 = NDArray.random(new int[]{1, 2});
        NDArray<Double> concat1 = NDArray.random(new int[]{2, 2});
        print("concat0", concat0);
        print(concat1);
        print(concat(concat0, concat1));
        print(Arrays.toString(concat0.get(slice(1, null)).shape));

        print(new NDArray<Double>(new int[]{0, 2, 9}), "wtf");

        print(mean(arange(0, 100, 1).reshape(new int[]{5, 5, -1}).astype(NDArray::Double), 1));

        print(abs(sub(random(new int[]{5, 2}), array(new Double[]{0.5}))));

        for (int[] index : NDArray.ndindex(new int[]{1, 5, 2})) {
            print(Arrays.toString(index));
        }
        print(NDArray.ndindex(new int[]{1, 5, 2}));

        print(NDArray.broadcast_shapes(new int[][]{{1, 2}, {3, 1}, {2, 3, 2}}));
        print(NDArray.broadcast_shapes());

        x1 = NDArray.arange(0, 9, 1).astype(NDArray::Double).reshape(new int[]{3, 3});
        x2 = NDArray.arange(0, 3, 1).astype(NDArray::Double);
        print("x1: " + x1);
        print("x2: " + x2);
        print("x1 - x2: " + sub(x1, x2));

        print("np.all([[True,False],[True,True]])", all(array(new Boolean[][]{{true, false}, {true, true}})));
        print("np.any([[True,False],[True,True]])", any(array(new Boolean[][]{{true, false}, {true, true}})));

        print("np.zeros((1,)).item()", zeros(new int[]{1}).item(0));

        print("np.random.random((3,2,1,4))");
        NDArray<Double> r1 = random(new int[]{3, 2, 1, 4});
        print("_view", _view);
        print(_view.shape, _view.shape, _view.shape);
        print("_view.item(new int[]{0})", _view.item(new int[]{5}));
        print("_view.item(new int[]{0})", _view.item(new int[]{5}));
        print(zeros(new int[]{1, 0}).size);
        print(r1);
        print("item", r1.item(new int[]{0, 1, 0, 2}));
        print("item", r1.item(new int[]{0, 1, 0, 2}));
        print(random(new int[]{1}).item(0));

        NDArray<Integer> ar = array(2);
        print("ar = np.array(2)", ar);
        ar.set(99);
        print("ar.set(99)", ar);
        print("ar.shape", ar.shape, ar.ndim, ar.size);
        print(ar.item(0));

        print(new NDArray<Double>(new int[]{0, 1}).size);

        view.fill(123.0);
        print("view.fill(123.0)", view, ndarray);

        print(copy);
//        copy.resize(new int[]{2,3});
        print(copy);

        print("ndarray", ndarray.size, ndarray.shape, ndarray);
        NDArray<Double> flatten = ndarray.flatten();
        print("ndarray.flatten()", flatten.size, flatten.shape, flatten);

        print("flatten()", array(new Double[]{7.5, 53.0}).get(index(1)));
        print("np.array(7.5).flatten()", array(7.5).flatten());
        print(array(7.5));

        print(x1, x1.base == null);
        print(x1.reshape(new int[]{-1}));

        print(NDArray
                .array(new Double[][]{{1.0, 2.0, 3.0}, {3.0, 1.0, 4.0}, {2.0, 3.0, 4.0}})
                .get(slice(), index(1))
                .get(slice(0, -1))
                .get(slice(null, null, -1))
                .reshape(new int[]{2, -1, 1, 1})

        );

        print(arange(50, 0, -3));

        NDArray<Integer> a1 = arange(10).reshape(new int[]{2, -1});
        NDArray<Integer> a2 = arange(20).get(slice(null, 10)).reshape(new int[]{2, -1});
        print(a1, a2, array_equal(a1, a2));

        print(array_equal(array(a1.shape), array(a2.shape)));
        print(array_equal(
                array(new Double[][]{{1.0, 2.0, 3.0}, {3.0, 1.0, 4.0}, {2.0, 3.0, 4.0}}),
                array(new Double[][]{{1.0, 2.0, 3.0}, {3.0, 1.0, 4.0}, {2.0, 3.0, 4.0}})
        ));

        print(array(a1.shape));
        print(equal(array(1), a2));
        print(equal(array(new int[]{1}), array(1)));


        NDArray<Double> ax = arange(18).astype(NDArray::Double);
        ax = power(ax, array(3).astype(NDArray::Double));
        ax = ax.reshape(new int[]{-1, 3, 2}).copy();
        ax = ax.get(slice(), slice(null, -1));
        ax.transpose().set(new int[]{0, 1, 0}, -99.);
        print(ax);
        print(ax.transpose());

        print();

        demo();
        indexing();
        _sum();
        _mean();
        _amax();
        _argmax();

        _advanced_index();
        _nonzero();
        _where();
        _broadcast_to();

        print("END");
    }
    static void _nonzero(){
        print("nonzero");
        print(">>>", "x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])");
        NDArray<Integer> x = array(new Integer[][]{{3, 0, 0}, {0, 4, 0}, {5, 6, 0}});

        print(">>>", "x");
        print(x);

        print(">>>", "np.nonzero(x)");
        NDArray<Integer>[] c = nonzero(x);
        NDArray<Integer> a = c[0];
        NDArray<Integer> b = c[1];
        print(a, b);

        print();
    }

    static void _broadcast_to(){
        print("broadcast_to");
        print(">>>","x = np.array([1, 2, 3])");
        NDArray<Integer> x=array(new int[]{1,2,3});

        print(">>>","np.broadcast_to(x, (3, 3))");
        print(broadcast_to(x, new int[]{3,3}));
        print();
    }

    static void _where(){
        print("where");
        print(">>>","a = np.arange(10)");
        NDArray<Double> a = arange(10).astype(NDArray::Double);

        print(">>>","a");
        print(a);

        print(">>>","np.where(a < 5, a, 10*a)");
        print(where(
                lt(a, array(5).astype(NDArray::Double)),
                a,
                mul(a, array(10.))
        ));

        print(">>>","np.where([[True, False], [True, True]],[[1, 2], [3, 4]],[[9, 8], [7, 6]])");
        print(where(
                array(new Boolean[][]{{true, false},{true,true}}),
                array(new Integer[][]{{1, 2},{3,4}}),
                array(new Integer[][]{{9, 8},{7,6}})
        ));

        print(">>>","a = np.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])");
        a = array(new Double[][]{{0.,1.,2.},{0.,2.,4.}, {0.,3.,6.}});

        print(">>>","np.where(a < 4, a, -1)");
        print(where(
                lt(a, array(4.)),
                a,
                array(-1.)
        ));
        print();
    }

    static void _advanced_index(){
        print("advanced_index");

        print(">>>","x = np.array([1., -1., -2., 3])");
        NDArray<Double> x = array(new Double[]{1., -1., -2., 3.});

        print(">>>", x);
        print(x);

        print(">>>", "x < 0");
        print(lt(x, array(0.0)));

        print(">>>", "x = np.array([[0, 1], [1, 1], [2, 2]])");
        x=array(new Double[][]{{0., 1.}, {1., 1.}, {2., 2.}});

        print(">>>", x);
        print(x);

        print(">>>", "x <= 1");
        print(lte(x, array(1.)));

        print();
    }

    static void demo() {
        print("demo");

        NDArray<Integer> a = arange(15).reshape(new int[]{3, 5});
        print("a = np.arange(15).reshape(3, 5)");
        print("a", a);
        print("a.shape", a.shape);
        print("a.ndim", a.ndim);
        print("a.size", a.size);
        NDArray<Integer> b = array(new Integer[]{6, 7, 8});
        print("b = np.array([6,7,8])");
        print("b", b);

        print();
    }

    static void _sum() {
        print("sum");
        print(">>>", "np.sum([0.5, 1.5])");
        print(sum(array(new double[]{0.5, 1.5})));

        print(">>>", "np.sum([[0, 1], [0, 5]])");
        print(sum(array(new double[][]{{0., 1.}, {0., 5.}})));

        print(">>>", "np.sum([[0, 1], [0, 5]], axis=0)");
        print(sum(array(new double[][]{{0., 1.}, {0., 5.}}), 0));

        print(">>>", "np.sum([[0, 1], [0, 5]], axis=1)");
        print(sum(array(new double[][]{{0., 1.}, {0., 5.}}), 1));

        print(">>>", "c = np.arange(2*3*4*5).reshape(2,3,4,5)");
        NDArray<Double> c = arange(2 * 3 * 4 * 5).astype(NDArray::Double).reshape(new int[]{2, 3, 4, 5});

        print(">>>", "np.sum(c)");
        print(sum(c));

        print(">>>", "np.sum(c, axis=2)");
        print(sum(c, 2));

        print(">>>", "np.sum(c, axis=(0,2))");
        print(sum(c, new int[]{0, 2}));

        print(">>>", "np.sum(c, axis=(0,1,2))");
        print(sum(c, new int[]{0, 1, 2}));
        print();
    }

    static void _amax() {
        print("amax");
        print(">>>", "a = np.array([[0, 1], [2, 3]])");
        NDArray<Double> a = array(new Double[][]{{0., 1.}, {2., 3.}});
        print(">>>", "np.amax(a)");
        print(amax(a));
        print(">>>", "np.amax(a, axis=0)");
        print(amax(a, 0));
        print(">>>", "np.amax(a, axis=1)");
        print(amax(a, 1));
        print();
    }

    static void _mean() {
        print("mean");
        print(">>>", "a = np.array([[1, 2], [3, 4]])");
        NDArray<Double> a = array(new Double[][]{{1., 2.}, {3., 4.}});
        print(">>>", "np.mean(a)");
        print(mean(a));
        print(">>>", "np.mean(a, axis=0)");
        print(mean(a, 0));
        print(">>>", "np.mean(a, axis=1)");
        print(mean(a, 1));
        print();
    }

    static void indexing() {
        print("indexing");

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
        for (NDArray<Double> i : a) {
            print(Math.pow(i.item(), 1 / 3.));
        }

        print(">>>", "def f(x,y):");
        print("...", "\treturn 10*x+y");
        print(">>>", "b = np.fromfunction(f,(5,4),dtype=int)");
        NDArray<Integer> b = fromfunction((int[] index) -> 10 * index[0] + index[1], new int[]{5, 4});
        print(">>>", "b");
        print(b);

        print(">>>", "b[2,3]");
        print(b.item(new int[]{2, 3}));

        print(">>>", "b[0:5, 1]");
        print(b.get(slice(0, 5), index(1)));

        print(">>>", "b[:,1]");
        print(b.get(slice(), index(1)));

        print(">>>", "b[1:3, :]");
        print(b.get(slice(1, 3), slice()));

        print(">>>", "b[-1]");
        print(b.get(index(-1)));

        print(">>>", "for row in b:");
        print("...", "\tprint(row)");
        for (NDArray<Integer> row : b) {
            print(row);
        }

        print(">>>", "for element in b.flat:");
        print("...", "\tprint(element)");
        for (Integer element : b.flat) {
            print(element);
        }
        print();
    }

    static void _argmax() {
        print("argmax");
        print(">>>", "a = np.arange(6).reshape(2,3) + 10");
        NDArray<Double> a = add(
                arange(6).reshape(new int[]{2, 3}).astype(NDArray::Double),
                array(10).astype(NDArray::Double)
        );

        print(">>>", "a");
        print(a);

        print(">>>", "np.argmax(a)");
        print(argmax(a));

        print(">>>", "np.argmax(a, axis=0)");
        print(argmax(a, 0));

        print(">>>", "np.argmax(a, axis=1)");
        print(argmax(a, 1));

        print();
    }

    public static void main(String[] args) {
        test();
    }
}
