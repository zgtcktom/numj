package com.numj;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiFunction;

public class Main {

    public static Index index(int ind) {
        return new Index(ind);
    }

    public static Slice slice() {
        return new Slice(null, null, 1);
    }

    public static Slice slice(Integer start, Integer stop) {
        return new Slice(start, stop, 1);
    }

    public static Slice slice(Integer start, Integer stop, Integer step) {
        return new Slice(start, stop, step);
    }

    static private abstract class Selection {
    }

    static private class Index extends Selection {
        int ind;

        Index(int ind) {
            this.ind = ind;
        }

        public Indexing indexing(int length) {
            return new Indexing(ind < 0 ? ind + length : ind);
        }

        public String toString() {
            return String.valueOf(ind);
        }
    }

    static private class Slice extends Selection {
        Integer start, stop, step;

        Slice(Integer start, Integer stop, Integer step) {
            this.start = start;
            this.stop = stop;
            this.step = step;
        }

        public Slicing slicing(int length) {
            Integer start = this.start,
                    stop = this.stop,
                    step = this.step;

            if (step == null) {
                step = 1;
            }

            int defStart = step < 0 ? length - 1 : 0;
            int defStop = step < 0 ? -1 : length;

            if (start == null) {
                start = defStart;
            } else {
                if (start < 0) start += length;
                if (start < 0) start = step < 0 ? -1 : 0;
                if (start >= length) start = step < 0 ? length - 1 : length;
            }

            if (stop == null) {
                stop = defStop;
            } else {
                if (stop < 0) stop += length;
                if (stop < 0) stop = step < 0 ? -1 : 0;
                if (stop >= length) stop = step < 0 ? length - 1 : length;
            }

            int sliceLength;
            if ((step == 0) || (step < 0 && stop >= start) || (step > 0 && start >= stop)) {
                sliceLength = 0;
            } else if (step < 0) {
                sliceLength = (stop - start + 1) / (step) + 1;
            } else {
                sliceLength = (stop - start - 1) / (step) + 1;
            }

            return new Slicing(start, stop, step, sliceLength);
        }

        public String toString() {
            return start + ":" + stop + ":" + step;
        }
    }

    static abstract private class Offset {
    }

    static private class Indexing extends Offset {
        private final int ind;

        public Indexing(int ind) {
            this.ind = ind;
        }

        public int get() {
            return ind;
        }

        public String toString() {
            return String.valueOf(ind);
        }
    }

    static private class Slicing extends Offset {
        final int start, stop, step;
        final int length;

        Slicing(int start, int stop, int step, int length) {
            this.start = start;
            this.stop = stop;
            this.step = step;
            this.length = length;
//            print(start+", "+stop+", "+step+", "+length);
        }

        public Slicing get(Slice slice) {
            Slicing sliced = slice.slicing(length);
            int start = this.start + sliced.start;
            int stop = this.start + sliced.stop;
            int step = this.step * sliced.step;
            return new Slicing(start, stop, step, sliced.length);
        }

        public Indexing get(Index index) {
            return new Indexing(get(index.indexing(length).get()));
        }

        public int get(int ind) {
            if (ind < 0) {
                ind += length;
            }
            return start + step * ind;
        }

        public String toString() {
            return start + ":" + stop + ":" + step;
        }
    }

    static class Operator {
        static Double add(Double a, Double b) {
            return a + b;
        }

        static Double sub(Double a, Double b) {
            return a - b;
        }

        static Double mul(Double a, Double b) {
            return a * b;
        }

        static Double div(Double a, Double b) {
            return a / b;
        }
    }

    static class Broadcast<T> implements Iterator<int[][]> {
        public final int[] shape;

        NDArray<T>[] ndarrays;

        private NDArray.IndexIterator indexIterator;
        private int[][] arrayIndices;

        @SafeVarargs
        public Broadcast(NDArray<T>... ndarrays) {
            this.ndarrays = ndarrays;
            this.shape = shape();
            if (shape != null) {
                indexIterator = new NDArray.IndexIterator(shape);
                arrayIndices = new int[ndarrays.length][];
                for (int i = 0; i < ndarrays.length; i++) {
                    arrayIndices[i] = new int[ndarrays[i].shape.length];
                }
            }
        }

        private int[] shape() {
            int ndim = 0;
            for (NDArray<T> ndarray : ndarrays) ndim = Math.max(ndarray.shape.length, ndim);
            int[] shape = new int[ndim];

            for (int i = 0; i < ndim; i++) {
                int m = ndim - 1 - i;
                shape[m] = 1;
                for (NDArray<T> ndarray : ndarrays) {
                    int n = ndarray.shape.length - 1 - i;
                    if (n >= 0) {
                        if (shape[m] == 1) shape[m] = ndarray.shape[n];
                        else if (ndarray.shape[n] != 1 && ndarray.shape[n] != shape[m]) {
                            return null;
                        }
                    }
                }
            }

            return shape;
        }

        public boolean hasNext() {
            return indexIterator.hasNext();
        }

        public int[][] next() {
            int[] current = indexIterator.next();
            for (int i = 0; i < ndarrays.length; i++) {
                int[] indices = arrayIndices[i];
                int[] shape = ndarrays[i].shape;
                for (int j = 0; j < indices.length; j++) {
                    int n = indices.length - 1 - j;
//                    print(n + ", " + (current.length -1 -j));
                    indices[n] = shape[n] == 1 ? 0 : current[current.length - 1 - j];
                }
            }
            return arrayIndices;
        }
    }

    static class NDArray<T> {
        public int[] shape;
        private final TypedArray<T> data;
        private Offset[] offsets;

        public NDArray(int[] shape) {
            this.data = new TypedArray<>(shape);
            this.offsets = null;
            this.shape = shape;
        }

        public NDArray(TypedArray<T> data, Offset[] offsets) {
            this.data = data;
            this.offsets = offsets;
            this.shape = shape();
        }

        public static <T> NDArray<T> array(Object[] arr) {
            List<Integer> shapeList = new ArrayList<>();
            shapeList.add(arr.length);

            Object[] nested = arr;
            while (nested[0] instanceof Object[]) {
                nested = (Object[]) nested[0];
                shapeList.add(nested.length);
            }

            int[] shape = new int[shapeList.size()];
            for (int i = 0; i < shape.length; i++) shape[i] = shapeList.get(i);

            NDArray<T> ndarray = new NDArray<>(shape);
            deepCopy(arr, ndarray, 0, 0);
            return ndarray;
        }

        @SuppressWarnings("unchecked")
        public static <T> void deepCopy(Object[] arr, NDArray<T> ndarray, int dim, int start) {
            for (int i = 0; i < arr.length; i++) {
                if (dim < ndarray.shape.length - 1) {
                    deepCopy((Object[]) arr[i], ndarray, dim + 1, start + i * ndarray.length(dim + 1));
                } else {
                    ndarray.data(start + i, (T) arr[i]);
                }
            }
        }

        static public <T> NDArray<T> apply(NDArray<T> a, NDArray<T> b, BiFunction<T, T, T> func) {
            Broadcast<T> bc = new Broadcast<>(a, b);
            if (bc.shape == null) return null;
            NDArray<T> out = new NDArray<>(bc.shape);
            for (int i = 0; bc.hasNext(); i++) {
                int[][] indices = bc.next();
                out.data(i, func.apply(a.getValue(indices[0]), b.getValue(indices[1])));
            }
            return out;
        }

        static public NDArray<Double> add(NDArray<Double> a, NDArray<Double> b) {
            return apply(a, b, Operator::add);
        }

        static public NDArray<Double> sub(NDArray<Double> a, NDArray<Double> b) {
            return apply(a, b, Operator::sub);
        }

        static public NDArray<Double> mul(NDArray<Double> a, NDArray<Double> b) {
            return apply(a, b, Operator::mul);
        }

        static public NDArray<Double> div(NDArray<Double> a, NDArray<Double> b) {
            return apply(a, b, Operator::div);
        }

        public NDArray<T> view() {
            return new NDArray<>(data, offsets);
        }

        public NDArray<T> copy() {
            NDArray<T> ndarray = new NDArray<>(shape);
            for (Iterator<int[]> it = indexIterator(); it.hasNext(); ) {
                int[] indices = it.next();
                ndarray.setValue(indices, getValue(indices));
            }
            return ndarray;
        }

        static public NDArray<Integer> arange(int start, int stop, int step){

            int length;
            if ((step == 0) || (step < 0 && stop >= start) || (step > 0 && start >= stop)) {
                length = 0;
            } else if (step < 0) {
                length = (stop - start + 1) / (step) + 1;
            } else {
                length = (stop - start - 1) / (step) + 1;
            }

            Integer[] data = new Integer[length];

            for (int i = start, j=0; i < stop; i+=step, j++) {
                data[j] = i;
            }

            return array(data);
        }

        public NDArray<T> reshape(int[] shape){
            NDArray<T> ndarray = copy();
            int unknown = -1;
            int size = ndarray.length(0);
            int rest = 1;
            int[] _shape = new int[shape.length];
            for (int i = 0; i < shape.length; i++) {
                int dim = shape[i];
                _shape[i] = dim;
                if(dim == -1){
                    if(unknown != -1) return null;
                    unknown = i;
                }else {
                    rest *= dim;
                }
            }
            if(unknown != -1){
                if(size%rest != 0) return null;
                _shape[unknown] = size/rest;
            }
//            print("" + size + ", "+rest);
//            print(Arrays.toString(ndarray.data.shape));
//            print(Arrays.toString(ndarray.data.data));
            ndarray.data.shape = _shape;
            ndarray.shape = ndarray.shape();
            return ndarray;
        }

        public T data(int index) {
            return data.get(index);
        }

        public void data(int index, T value) {
            data.set(index, value);
        }

        public int length(int dim) {
            if (dim >= shape.length) return 1;
            return shape[dim] * length(dim + 1);
        }

        public NDArray<T> get(Selection... selections) {
            if (selections.length > shape.length) return null;
            if (offsets == null) offsets = offsets();
            List<Offset> offsetList = new ArrayList<>();

            for (int i = 0, j = 0; i < offsets.length; i++) {
                if (offsets[i] instanceof Slicing) {
                    Slicing slicing = (Slicing) offsets[i];
                    if (j < selections.length) {
                        if (selections[j] instanceof Slice) {
                            // Slicing, slice
                            Slice slice = (Slice) selections[j];
                            offsetList.add(slicing.get(slice));
                        } else {
                            // Slicing, index
                            Index index = (Index) selections[j];
                            offsetList.add(slicing.get(index));
                        }
                        j++;
                    } else {
                        // Slicing, copy
                        offsetList.add(slicing);
                    }
                } else {
                    // Indexing, copy
                    Indexing indexing = (Indexing) offsets[i];
                    offsetList.add(indexing);
                }
            }

            Offset[] offsets = offsetList.toArray(new Offset[0]);
            return new NDArray<>(data, offsets);
        }

        private int[] shape() {
            if (offsets == null) return data.shape;
            List<Integer> shapeList = new ArrayList<>();
            for (Offset offset : offsets) {
                if (offset instanceof Slicing) {
                    Slicing slicing = (Slicing) offset;
                    shapeList.add(slicing.length);
                }
            }
            return toArray(shapeList);
        }

        private static int[] toArray(List<Integer> list){
            int[] array = new int[list.size()];
            for (int i = 0; i < array.length; i++) {
                array[i] = list.get(i);
            }
            return array;
        }

        private Offset[] offsets() {
            int N = data.shape.length;
            Offset[] offsets = new Offset[N];
            for (int i = 0; i < N; i++) {
                offsets[i] = slice().slicing(data.shape[i]);
            }
            return offsets;
        }

        private int getIndex(int[] indices) {
            if (indices.length != shape.length) return -1;

            int ind = 0;
            int length = 1;
            for (int i = data.shape.length - 1, j = indices.length - 1; i >= 0; i--) {
                int n;
                if (offsets != null) {
                    if (offsets[i] instanceof Slicing) {
                        Slicing slicing = (Slicing) offsets[i];
                        n = slicing.get(indices[j]);
                        j--;
                    } else {
                        Indexing indexing = (Indexing) offsets[i];
                        n = indexing.get();
                    }
                } else {
                    n = indices[i];
                }
                ind += n * length;
                length *= data.shape[i];
            }

            return ind;
        }

        public T getValue(int[] indices) {
            int ind = getIndex(indices);
            if (ind == -1) return null;

            return data(ind);
        }

        public void setValue(int[] indices, T value) {
            int ind = getIndex(indices);
            if (ind == -1) return;

            data(ind, value);
        }

        public String toString() {
            if (shape.length == 0) return data(0).toString();

            StringBuilder string = new StringBuilder();
            if (shape.length == 1) {
                int[] indices = new int[1];
                for (int i = 0; i < shape[0]; i++) {
                    if (string.length() > 0) string.append(", ");
                    indices[0] = i;
                    string.append(getValue(indices));
                }
            } else {
                for (int i = 0; i < shape[0]; i++) {
                    if (string.length() > 0) string.append(", ");
                    string.append(get(index(i)));
                }
            }
            return "[" + string + "]";
        }

        public Iterator<int[]> indexIterator() {
            return new IndexIterator(shape);
        }

        static private class TypedArray<T> {
            public int[] shape;
            private final Object[] data;

            public TypedArray(int[] shape) {
                int length = 1;
                for (int n : shape) length *= n;

                this.shape = shape;
                this.data = new Object[length];
            }

            @SuppressWarnings("unchecked")
            public T get(int index) {
                return (T) data[index];
            }

            public void set(int index, T value) {
                data[index] = value;
            }
        }

        static private class IndexIterator implements Iterator<int[]> {
            private final int[] shape;
            private final int[] indices;
            private final int[] out;
            private boolean hasNext;

            public IndexIterator(int[] shape) {
                this.shape = shape;
                this.indices = new int[shape.length];
                this.hasNext = true;

                this.out = new int[shape.length];
            }

            public boolean hasNext() {
                return hasNext;
            }

            public int[] next() {
                int carry = 1;
                for (int i = indices.length - 1; i >= 0; i--) {
                    out[i] = indices[i];

                    indices[i] += carry;
                    carry = 0;
                    if (indices[i] >= shape[i]) {
                        indices[i] -= shape[i];
                        carry = 1;
                    }
                }
                if (carry != 0) hasNext = false;
                return out;
            }
        }
    }

    static void print(Object... args){
        StringBuilder string = new StringBuilder();
        for (Object arg:args) {
            if(string.length() != 0) string.append(" ");
            string.append(arg.toString());
        }
        System.out.println(string.toString());
    }

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


        NDArray<Double> x1 = NDArray.array(new Double[][]{
                {0.0, 1.0, 2.0},
                {3.0, 4.0, 5.0},
                {6.0, 7.0, 8.0}
        });

        NDArray<Double> x2 = NDArray.array(new Double[][]{
                {0.0, 1.0, 2.0},
                {3.0, -1.0, 5.0},
                {6.0, 7.0, 8.0}
        });

        print(Arrays.toString(x1.data.data));
        print("x1: " + x1);
        print("x2: " + x2);

        NDArray<Double> out = NDArray.add(x1, x2);

        print(out.toString());

        NDArray<Double> x = NDArray.array(new Double[]{1.0, 2.0, 3.0});
        NDArray<Double> y = NDArray.array(new Double[][]{{4.0}, {5.0}, {6.0}});
        print("x: " + x);
        print("y: " + y);

        print("x + y: " + NDArray.add(x, y));
        print("x + 2: " + NDArray.add(x, NDArray.array(new Double[]{2.0})));
        print("2 + y: " + NDArray.add(NDArray.array(new Double[]{2.0}), y));

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

        print(NDArray.add(NDArray.mul(x, y), NDArray.div(NDArray.sub(y, x), x)).toString());

        NDArray<Integer> a = NDArray.arange(0, 6, 1).reshape(new int[]{3, 2});
        print(a);
        a = NDArray.array(new Integer[][]{{1,2,3},{4,5,6}});
        a = a.reshape(new int[]{3, -1});
        print(a);

        print("END");
    }

    public static void main(String[] args) {
        test();
    }
}
