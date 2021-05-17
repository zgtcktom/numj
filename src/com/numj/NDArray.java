package com.numj;


import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;


class NDArray<T> {

    private static final Random rand = new Random();
    public final TypedArray<T> data;
    public final int[] shape;
    public final int ndim;
    public final int size;
    public final NDArray<T> base;
    private final int[] strides;
    private final Offset[] offsets;

    public NDArray(int[] shape) {
        this(shape, null);
    }

    private NDArray(int[] shape, TypedArray<T> data) {
        this.shape = shape;
        ndim = shape.length;
        strides = new int[ndim];
        int size = 1;
        for (int i = strides.length - 1; i >= 0; i--) {
            strides[i] = size;
            size *= shape[i];
        }
        this.size = size;
        if (data == null) data = new TypedArray<>(size);
        this.data = data;
        base = null;
        offsets = new Offset[ndim];
        for (int i = 0; i < offsets.length; i++) {
            offsets[i] = new Slicing(0, shape[i], 1, shape[i]);
        }
    }

    private NDArray(NDArray<T> base, Offset[] offsets, int[] strides) {
        this.offsets = offsets;
        this.base = base.base != null ? base.base : base;
        this.strides = strides;
        data = base.data;

        int ndim = 0;
        for (Offset offset : offsets) ndim += offset.ndim;
        int size = 1;
        int[] shape = new int[ndim];
        for (int i = 0, j = 0; i < offsets.length; i++) {
            if (offsets[i].ndim != 0) {
                shape[j] = offsets[i].length;
                size *= offsets[i].length;
                j++;
            }
        }
        this.ndim = ndim;
        this.shape = shape;
        this.size = size;
    }

    @SuppressWarnings("unchecked")
    public static <T> void deepCopy(Object[] arr, NDArray<T> ndarray, int dim, int start) {
        for (int i = 0; i < arr.length; i++) {
            if (dim < ndarray.ndim - 1) {
                deepCopy(
                        (Object[]) arr[i],
                        ndarray,
                        dim + 1,
                        start + i * ndarray.strides[dim]
                );
            } else {
                ndarray.itemset(start + i, (T) arr[i]);
            }
        }
    }

    public static NDArray<Integer> array(int[] arr) {
        int n = arr.length;
        NDArray<Integer> out = new NDArray<>(new int[]{n});
        for (int i = 0; i < n; i++) {
            out.itemset(i, arr[i]);
        }
        return out;
    }

    public static NDArray<Integer> array(int[][] arr) {
        int n = arr.length, m = arr[0].length;
        NDArray<Integer> out = new NDArray<>(new int[]{n, m});
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                out.itemset(i * m + j, arr[i][j]);
            }
        }
        return out;
    }

    public static NDArray<Double> array(double[] arr) {
        int n = arr.length;
        NDArray<Double> out = new NDArray<>(new int[]{n});
        for (int i = 0; i < n; i++) {
            out.itemset(i, arr[i]);
        }
        return out;
    }

    public static NDArray<Double> array(double[][] arr) {
        int n = arr.length, m = arr[0].length;
        NDArray<Double> out = new NDArray<>(new int[]{n, m});
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                out.itemset(i * m + j, arr[i][j]);
            }
        }
        return out;
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

    private static int getLength(int start, int stop, int step) {
        int length;
        if ((step == 0) || (step < 0 && stop >= start) || (step > 0 && start >= stop)) {
            length = 0;
        } else if (step < 0) {
            length = (stop - start + 1) / step + 1;
        } else {
            length = (stop - start - 1) / step + 1;
        }
        return length;
    }

    public static NDArray<Integer> arange(int stop) {
        return arange(0, stop, 1);
    }

    public static NDArray<Integer> arange(int start, int stop) {
        return arange(start, stop, 1);
    }

    public static NDArray<Integer> arange(int start, int stop, int step) {
        int length = getLength(start, stop, step);
        NDArray<Integer> ndarray = new NDArray<>(new int[]{length});
        for (int i = 0, j = start; i < length; i++, j += step) {
            ndarray.itemset(i, j);
        }
        return ndarray;
    }

    public static int prod(int[] arr) {
        int out = 1;
        for (int val : arr) {
            out *= val;
        }
        return out;
    }

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

    public static <T> NDArray<T> array(T value) {
        NDArray<T> out = new NDArray<>(new int[]{});
        out.itemset(0, value);
        return out;
    }

    static public <T, E> NDArray<E> apply(NDArray<T> a, NDArray<T> b, BiFunction<T, T, E> func) {
        Broadcast broadcast = broadcast(a, b);
        NDArray<E> out = new NDArray<>(broadcast.shape);
        int i = 0;
        for (int[][] indices : broadcast) {
            out.itemset(i, func.apply(a.item(indices[0]), b.item(indices[1])));
            i++;
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

    static public NDArray<Boolean> gt(NDArray<Double> a, NDArray<Double> b) {
        return apply(a, b, Operator::gt);
    }

    static public NDArray<Boolean> gte(NDArray<Double> a, NDArray<Double> b) {
        return apply(a, b, Operator::gte);
    }

    static public NDArray<Boolean> lt(NDArray<Double> a, NDArray<Double> b) {
        return apply(a, b, Operator::lt);
    }

    static public NDArray<Boolean> lte(NDArray<Double> a, NDArray<Double> b) {
        return apply(a, b, Operator::lte);
    }

    static public NDArray<Boolean> eq(NDArray<Double> a, NDArray<Double> b) {
        return apply(a, b, Operator::eq);
    }

    static public NDArray<Boolean> neq(NDArray<Double> a, NDArray<Double> b) {
        return apply(a, b, Operator::neq);
    }

    static public NDArray<Double> zeros(int[] shape) {
        NDArray<Double> ndarray = new NDArray<>(shape);
        for (int[] index : ndindex(ndarray.shape)) {
            ndarray.itemset(index, 0.0);
        }
        return ndarray;
    }

    static public Boolean all(NDArray<Boolean> ndarray) {
        for (int[] index : ndindex(ndarray.shape)) {
            if (!ndarray.item(index)) return false;
        }
        return true;
    }

    static public Boolean any(NDArray<Boolean> ndarray) {
        for (int[] index : ndindex(ndarray.shape)) {
            if (ndarray.item(index)) return true;
        }
        return false;
    }

    static public NDArray<Double> ones(int[] shape) {
        NDArray<Double> ndarray = new NDArray<>(shape);
        for (int[] index : ndindex(ndarray.shape)) {
            ndarray.itemset(index, 1.0);
        }
        return ndarray;
    }

    static public NDArray<Double> random(int[] shape) {
        NDArray<Double> ndarray = new NDArray<>(shape);
        for (int[] index : ndindex(ndarray.shape)) {
            ndarray.itemset(index, rand.nextDouble());
        }
        return ndarray;
    }

    static public Double amin(NDArray<Double> ndarray) {
        Double min = null;
        for (int[] index : ndindex(ndarray.shape)) {
            Double value = ndarray.item(index);
            if (min == null) min = value;
            else min = Math.min(min, value);
        }
        return min;
    }

    static public NDArray<Double> nan_to_num(NDArray<Double> ndarray) {
        NDArray<Double> out = ndarray.copy();
        for (int[] index : ndindex(ndarray.shape)) {
            Double value = ndarray.item(index);
            if (value == null || Double.isNaN(value))
                value = 0.0;
            out.itemset(index, value);
        }
        return out;
    }

    static public NDArray<Double> amin(NDArray<Double> ndarray, int axis) {
        int[] shape = new int[ndarray.shape.length - 1];
        Selection[] selections = new Selection[ndarray.shape.length];
        for (int i = 0, j = 0; i < ndarray.shape.length; i++) {
            if (axis != i) {
                shape[j] = ndarray.shape[i];
                j++;
            }
        }
        NDArray<Double> out = new NDArray<>(shape);

        int[] outIndices = new int[out.shape.length];
        for (int[] index : ndindex(ndarray.shape)) {
            if (index[axis] != 0) continue;
            for (int i = 0, j = 0; i < selections.length; i++) {
                if (i == axis) selections[i] = slice();
                else {
                    selections[i] = index(index[i]);
                    outIndices[j] = index[i];
                    j++;
                }
            }
            out.itemset(outIndices, amin(ndarray.get(selections)));
        }
        return out;
    }

    static public NDArray<Double> mean(NDArray<Double> ndarray, int axis) {
        int[] shape = new int[ndarray.shape.length - 1];
        Selection[] selections = new Selection[ndarray.shape.length];
        for (int i = 0, j = 0; i < ndarray.shape.length; i++) {
            if (axis != i) {
                shape[j] = ndarray.shape[i];
                j++;
            }
        }
        NDArray<Double> out = new NDArray<>(shape);

        int[] outIndices = new int[out.shape.length];
        for (int[] index : ndindex(ndarray.shape)) {
            if (index[axis] != 0) continue;
            for (int i = 0, j = 0; i < selections.length; i++) {
                if (i == axis) selections[i] = slice();
                else {
                    selections[i] = index(index[i]);
                    outIndices[j] = index[i];
                    j++;
                }
            }
            out.itemset(outIndices, mean(ndarray.get(selections)));
        }
        return out;
    }

    static public Double mean(NDArray<Double> ndarray) {
        Double total = 0.0;
        int count = 0;
        for (int[] index : ndindex(ndarray.shape)) {
            Double value = ndarray.item(index);
            total += value;
            count++;
        }
        if (count > 0)
            return total / count;
        return null;
    }

    static public NDArray<Double> abs(NDArray<Double> ndarray) {
        NDArray<Double> out = new NDArray<>(ndarray.shape);
        for (int[] index : ndindex(ndarray.shape)) {
            Double value = ndarray.item(index);
            out.itemset(index, Math.abs(value));
        }
        return out;
    }

    static public NDArray<Double> amax(NDArray<Double> ndarray, int axis) {
        int[] shape = new int[ndarray.shape.length - 1];
        Selection[] selections = new Selection[ndarray.shape.length];
        for (int i = 0, j = 0; i < ndarray.shape.length; i++) {
            if (axis != i) {
                shape[j] = ndarray.shape[i];
                j++;
            }
        }
        NDArray<Double> out = new NDArray<>(shape);

        int[] outIndices = new int[out.shape.length];
        for (int[] index : ndindex(ndarray.shape)) {
            if (index[axis] != 0) continue;
            for (int i = 0, j = 0; i < selections.length; i++) {
                if (i == axis) selections[i] = slice();
                else {
                    selections[i] = index(index[i]);
                    outIndices[j] = index[i];
                    j++;
                }
            }
            out.itemset(outIndices, amax(ndarray.get(selections)));
        }
        return out;
    }

    static public Double amax(NDArray<Double> ndarray) {
        Double max = null;
        for (int[] index : ndindex(ndarray.shape)) {
            Double value = ndarray.item(index);
            if (max == null) max = value;
            else max = Math.max(max, value);
        }
        return max;
    }

    static public int argmax(NDArray<Double> ndarray) {
        Double max = null;
        int ind = -1;
        int i = 0;
        for (int[] index : ndindex(ndarray.shape)) {
            Double value = ndarray.item(index);
            if (max == null || value > max) {
                max = value;
                ind = i;
            }
            i++;
        }
        return ind;
    }

    @SafeVarargs
    static public <T> NDArray<T> concat(NDArray<T>... ndarrays) {
        int length = 0;
        for (NDArray<T> ndarray : ndarrays) {
            if (ndarray.shape.length != ndarrays[0].shape.length) return null;
            for (int i = 1; i < ndarrays[0].shape.length; i++) {
                if (ndarray.shape[i] != ndarrays[0].shape[i]) return null;
            }
            length += ndarray.shape[0];
        }
        int[] shape = new int[ndarrays[0].shape.length];
        shape[0] = length;
        for (int i = 1; i < ndarrays[0].shape.length; i++) {
            shape[i] = ndarrays[0].shape[i];
        }
        NDArray<T> out = new NDArray<>(shape);
        int start = 0;
        for (NDArray<T> ndarray : ndarrays) {
            int stop = start + ndarray.shape[0];
            out.get(slice(start, stop)).set(ndarray);
            start = stop;
        }
        if (start != out.shape[0]) return null;
        return out;
    }

    static void print(Object... args) {
        StringBuilder string = new StringBuilder();
        for (Object arg : args) {
            if (string.length() != 0) string.append(" ");
            String str;
            if (arg instanceof int[]) {
                str = Arrays.toString((int[]) arg);
            } else {
                str = arg.toString();
            }
            string.append(str);
        }
        System.out.println(string.toString());
    }

    static Double Double(Float n) {
        return (double) (float) n;
    }

    static Double Double(Integer n) {
        return (double) (int) n;
    }

    static Float Float(Double n) {
        return (float) (double) n;
    }

    static Float Float(Integer n) {
        return (float) (int) n;
    }

    static Integer Integer(Double n) {
        return (int) (double) n;
    }

    static Integer Integer(Float n) {
        return (int) (float) n;
    }

    static Integer Integer(Boolean n) {
        return n ? 1 : 0;
    }

    static Boolean Boolean(Integer n) {
        return n != 0;
    }

    static NdIndex ndindex(int[] shape) {
        return new NdIndex(shape);
    }

    static public int[] broadcast_shapes(int[]... shapes) {
        int ndim = 0;
        for (int[] shape : shapes) ndim = Math.max(shape.length, ndim);

        int[] out = new int[ndim];
        for (int i = 0; i < ndim; i++) {
            int m = ndim - 1 - i;
            out[m] = 1;
            for (int[] shape : shapes) {
                int n = shape.length - 1 - i;
                if (n < 0) continue;
                if (out[m] == 1) out[m] = shape[n];
                else if (shape[n] != 1 && shape[n] != out[m]) return null;
            }
        }
        return out;
    }

    public static Broadcast broadcast(NDArray<?>... ndarrays) {
        return new Broadcast(ndarrays);
    }

    public static <T> Boolean array_equal(T a, T b) {
        return a.equals(b);
    }

    public static Boolean array_equal(int[] a, int[] b) {
        return Arrays.equals(a, b);
    }

    public static Boolean array_equal(Object[] a, Object[] b) {
        return array_equal(array(a), array(b));
    }

    public static <T> Boolean array_equal(NDArray<T> a, NDArray<T> b) {
        if (a == b) return true;
        if (a.ndim != b.ndim || !Arrays.equals(a.shape, b.shape)) return false;
        for (int[] index : ndindex(a.shape)) {
            if (!array_equal(a.item(index), b.item(index))) return false;
        }
        return true;
    }

    public static <T> NDArray<Boolean> equal(NDArray<T> a, NDArray<T> b) {
        Broadcast broadcast = broadcast(a, b);
        NDArray<Boolean> out = new NDArray<>(broadcast.shape);
        int i = 0;
        for (int[][] indices : broadcast) {
            out.itemset(i, array_equal(a.item(indices[0]), b.item(indices[1])));
            i++;
        }
        return out;
    }

    public T item(int index) {
        return data.get(index);
    }

    public T item(int[] index) {
        return data.get(itemindex(index));
    }

    private int itemindex(int[] index) {
        if (index.length != ndim) throw new ArrayIndexOutOfBoundsException();
        int ind = 0;
        for (int i = 0, j = 0; i < offsets.length; i++) {
            int n;
            if (offsets[i].ndim == 0) { // Indexing
                n = offsets[i].get();
            } else { // Slicing
                n = offsets[i].get(index[j]);
                j++;
            }
            ind += n * strides[i];
        }
        return ind;
    }

    public void itemset(int index, T value) {
        data.set(index, value);
    }

    public void itemset(int[] index, T value) {
        data.set(itemindex(index), value);
    }

    public NDArray<T> get(Selection... selections) {
        if (selections.length > shape.length) throw new RuntimeException();
        List<Offset> offsetList = new ArrayList<>();

        for (int i = 0, j = 0; i < offsets.length; i++) {
            Offset offset = offsets[i];
            if (offset.ndim != 0 && j < selections.length) {
                offset = offset.get(selections[j++]);
            }
            offsetList.add(offset);
        }
        Offset[] offsets = offsetList.toArray(new Offset[0]);
        return new NDArray<>(this, offsets, strides);
    }

    public String toString() {
        if (ndim == 0) return "" + item(new int[0]);

        StringBuilder string = new StringBuilder();
        if (ndim == 1) {
            int[] index = new int[1];
            for (int i = 0; i < shape[0]; i++) {
                if (string.length() > 0) string.append(", ");
                index[0] = i;
                string.append(item(index));
            }
        } else {
            for (int i = 0; i < shape[0]; i++) {
                if (string.length() > 0) string.append(", ");
                string.append(get(new Index(i)));
            }
        }
        return "[" + string + "]";
    }

    public <E> NDArray<E> astype(Function<T, E> valueOf) {
        NDArray<E> ndarray = new NDArray<>(shape);
        for (int[] index : ndindex(shape)) {
            ndarray.itemset(index, valueOf.apply(item(index)));
        }
        return ndarray;
    }

    public NDArray<T> view() {
        return new NDArray<>(this, offsets, strides);
    }

    public NDArray<T> copy() {
        NDArray<T> out = new NDArray<>(shape);
        out.set(this);
        return out;
    }

    public NDArray<T> reshape(int[] shape) {
        int unknown = -1;
        int rest = 1;
        shape = Arrays.copyOf(shape, shape.length);
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] == -1) {
                if (unknown != -1) throw new RuntimeException();
                unknown = i;
            } else {
                rest *= shape[i];
            }
        }
        if (rest == 0) throw new RuntimeException();
        if (unknown != -1) {
            if (size % rest != 0) throw new RuntimeException();
            shape[unknown] = size / rest;
        }
        return new NDArray<>(shape, copy().data);
    }

    public void fill(T value) {
        for (int[] index : ndindex(shape)) {
            itemset(index, value);
        }
    }

    public NDArray<T> flatten() {
        NDArray<T> out = new NDArray<>(new int[]{size});
        if (size == 1) {
            out.itemset(0, item(0));
        } else {
            int i = 0;
            for (int[] index : ndindex(shape)) {
                out.itemset(i, item(index));
                i++;
            }
        }
        return out;
    }

    public void set(int[] index, T value) {
        itemset(index, value);
    }

    public void set(Selection[] selections, T value) {
        get(selections).set(value);
    }

    public void set(T value) {
        if (size == 1) {
            itemset(0, value);
        } else {
            for (int[] index : ndindex(shape)) {
                itemset(index, value);
            }
        }
    }

    public void set(Selection[] selections, NDArray<T> src) {
        get(selections).set(src);
    }

    public void set(NDArray<T> src) {
        if (Arrays.equals(shape, src.shape)) {
            for (int[] index : ndindex(shape)) {
                itemset(index, src.item(index));
            }
        } else {
            Broadcast broadcast = broadcast(this, src);
            if (!Arrays.equals(broadcast.shape, shape)) throw new RuntimeException();
            for (int[][] indices : broadcast) {
                itemset(indices[0], src.item(indices[1]));
            }
        }
    }

    static abstract class Selection {
    }

    static public class Index extends Selection {
        int index;

        public Index(int index) {
            this.index = index;
        }

        public Indexing indexing(int length) {
            return new Indexing(index < 0 ? index + length : index);
        }

        public String toString() {
            return String.valueOf(index);
        }
    }

    static public class Slice extends Selection {
        Integer start, stop, step;

        public Slice(Integer start, Integer stop, Integer step) {
            this.start = start;
            this.stop = stop;
            this.step = step;
        }

        public Slicing slicing(int length) {
            Integer start = this.start, stop = this.stop, step = this.step;

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

    private static abstract class Offset {
        // Slicing {start, stop, step}
        // Indexing {index}
        public int ndim;
        public int length;

        public Offset(int ndim, int length) {
            this.ndim = ndim;
            this.length = length;
        }

        public Offset get(Selection selection) {
            return null;
        }

        public int get(int index) {
            return 0;
        }

        public int get() {
            return 0;
        }
    }

    private static class Indexing extends Offset {
        private final int index;

        public Indexing(int index) {
            super(0, 1);
            this.index = index;
        }

        public int get() {
            return index;
        }

        public String toString() {
            return String.valueOf(index);
        }
    }

    private static class Slicing extends Offset {
        private final int start, stop, step;

        public Slicing(int start, int stop, int step, int length) {
            super(1, length);
            this.start = start;
            this.stop = stop;
            this.step = step;
        }

        public int get(int index) {
            if (index < 0) {
                index += length;
            }
            return start + step * index;
        }

        public String toString() {
            return start + ":" + stop + ":" + step;
        }

        public Offset get(Selection selection) {
            if (selection instanceof Slice) return get((Slice) selection);
            return get((Index) selection);
        }

        public Slicing get(Slice slice) {
            Slicing slicing = slice.slicing(length);
            return new Slicing(
                    start + slicing.start,
                    start + slicing.stop,
                    step * slicing.step,
                    slicing.length
            );
        }

        public Indexing get(Index index) {
            return new Indexing(get(index.indexing(length).get()));
        }
    }

    private static class TypedArray<T> {
        public final Object[] buffer;

        public TypedArray(int length) {
            buffer = new Object[length];
        }

        @SuppressWarnings("unchecked")
        public T get(int index) {
            return (T) buffer[index];
        }

        public void set(int index, T value) {
            buffer[index] = value;
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

        static Boolean gt(Double a, Double b) {
            return a > b;
        }

        static Boolean gte(Double a, Double b) {
            return a >= b;
        }

        static Boolean lt(Double a, Double b) {
            return a < b;
        }

        static Boolean lte(Double a, Double b) {
            return a <= b;
        }

        static Boolean eq(Double a, Double b) {
            return a.equals(b);
        }

        static Boolean neq(Double a, Double b) {
            return !a.equals(b);
        }
    }

    private static class Broadcast implements Iterable<int[][]> {
        public final int ndim;
        public final int[] shape;
        private final NDArray<?>[] ndarrays;

        public Broadcast(NDArray<?>... ndarrays) {
            int[][] shapes = new int[ndarrays.length][];
            for (int i = 0; i < shapes.length; i++) {
                shapes[i] = ndarrays[i].shape;
            }
            this.ndarrays = ndarrays;
            this.shape = broadcast_shapes(shapes);
            assert this.shape != null;
            this.ndim = this.shape.length;
        }

        public Iterator<int[][]> iterator() {
            return new BroadcastIterator(this);
        }

        private static class BroadcastIterator implements Iterator<int[][]> {
            private final Broadcast broadcast;
            private final Iterator<int[]> iterator;
            private final int[][] out;

            public BroadcastIterator(Broadcast broadcast) {
                this.broadcast = broadcast;

                this.iterator = ndindex(broadcast.shape).iterator();
                this.out = new int[broadcast.ndarrays.length][];
                for (int i = 0; i < broadcast.ndarrays.length; i++) {
                    this.out[i] = new int[broadcast.ndarrays[i].ndim];
                }
            }

            public boolean hasNext() {
                return iterator.hasNext();
            }

            public int[][] next() {
                int[] value = iterator.next();
                for (int i = 0; i < broadcast.ndarrays.length; i++) {
                    int[] index = out[i];
                    int[] shape = broadcast.ndarrays[i].shape;
                    for (int j = 0; j < index.length; j++) {
                        int n = index.length - 1 - j;
                        index[n] = shape[n] == 1 ? 0 : value[value.length - 1 - j];
                    }
                }
                return out;
            }
        }
    }

    public static class NdIndex implements Iterable<int[]> {
        private final int[] shape;

        public NdIndex(int[] shape) {
            this.shape = shape;
        }

        public Iterator<int[]> iterator() {
            return new NdIndexIterator(this);
        }

        private static class NdIndexIterator implements Iterator<int[]> {
            private final NdIndex ndindex;
            private final int[] index;
            private final int[] out;
            private boolean hasNext;

            public NdIndexIterator(NdIndex ndindex) {
                this.ndindex = ndindex;
                int[] shape = ndindex.shape;
                this.index = new int[shape.length];
                this.out = new int[shape.length];

                if (shape.length > 0) {
                    int size = shape[0];
                    for (int i = 1; i < shape.length; i++) size *= shape[i];
                    hasNext = size != 0;
                } else {
                    hasNext = true;
                }
            }

            public boolean hasNext() {
                return hasNext;
            }

            public int[] next() {
                if (out.length == 0) {
                    hasNext = false;
                    return out;
                }
                int carry = 1;
                for (int i = index.length - 1; i >= 0; i--) {
                    out[i] = index[i];

                    index[i] += carry;
                    carry = 0;
                    if (index[i] >= ndindex.shape[i]) {
                        index[i] -= ndindex.shape[i];
                        carry = 1;
                    }
                }
                if (carry != 0) hasNext = false;
                return out;
            }
        }
    }
}