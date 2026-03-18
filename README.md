# JavaMathLib

A lightweight Java mathematics library providing matrix algebra, vector maths, Perlin noise generation, and a collection of general-purpose numeric utilities written by Mario Gianota. The library is designed with machine learning applications in mind, but is broadly useful for simulations, procedural graphics, and any numerically intensive Java project.

---

## Table of Contents

- [Classes at a Glance](#classes-at-a-glance)
- [Getting Started](#getting-started)
- [Matrix](#matrix)
  - [Creating a Matrix](#creating-a-matrix)
  - [Arithmetic Operations](#arithmetic-operations)
  - [Matrix Multiplication](#matrix-multiplication)
  - [Transpose and Inverse](#transpose-and-inverse)
  - [Element-wise Mapping](#element-wise-mapping)
  - [Utility Methods](#utility-methods)
- [Vector](#vector)
  - [Creating a Vector](#creating-a-vector)
  - [Arithmetic](#arithmetic)
  - [Geometric Operations](#geometric-operations)
- [MathUtils](#mathutils)
  - [Clamping and Mapping](#clamping-and-mapping)
  - [Interpolation](#interpolation)
  - [Random Numbers](#random-numbers)
  - [Perlin Noise](#perlin-noise)
- [Function Interface](#function-interface)
- [Known Bug: `Matrix.inverse()` Mutates the Original](#known-bug-matrixinverse-mutates-the-original)

---

## Classes at a Glance

| Class | Description |
|---|---|
| `Matrix` | Full-featured 2D matrix with ML-oriented operations |
| `Vector` | 3D (and 2D) vector maths |
| `MathUtils` | Static utility methods for common numeric operations |
| `PerlinNoise` | Ken Perlin's classic noise algorithm (3D) |
| `Function` | Functional interface for element-wise matrix mapping |
| `NonInvertibleMatrixException` | Checked exception thrown when matrix inversion fails |

---

## Getting Started

All classes are in the default package. Copy the `.java` files into your project and compile together:

```bash
javac *.java
```

No external dependencies are required.

---

## Matrix

The `Matrix` class wraps a `double[][]` array and exposes a rich set of operations suitable for neural networks and other ML workloads.

### Creating a Matrix

```java
// Empty matrix — populate later with readData()
Matrix a = new Matrix();

// Pre-sized matrix, all elements initialised to 0
Matrix b = new Matrix(3, 3);

// Load data from a 2D array
double[][] data = { {1, 2, 3}, {4, 5, 6} };
a.readData(data);

// Create a column-vector matrix from a 1D array
double[] values = {1.0, 2.0, 3.0};
Matrix col = Matrix.fromArray(values);
// Result:
// 1.0
// 2.0
// 3.0
```

### Arithmetic Operations

All arithmetic operations return a **new** `Matrix` and leave the original unchanged.

```java
double[][] dataA = { {1, 2}, {3, 4} };
double[][] dataB = { {5, 6}, {7, 8} };

Matrix a = new Matrix(); a.readData(dataA);
Matrix b = new Matrix(); b.readData(dataB);

// Matrix + Matrix
Matrix sum = a.add(b);
// [ 6,  8]
// [10, 12]

// Matrix + scalar
Matrix shifted = a.add(10.0);
// [11, 12]
// [13, 14]

// Matrix - Matrix
Matrix diff = b.subtract(a);
// [4, 4]
// [4, 4]

// Matrix - scalar
Matrix reduced = a.subtract(1.0);
// [0, 1]
// [2, 3]

// Scalar multiplication / division
Matrix scaled  = a.multiply(2.0);   // all elements * 2
Matrix divided = a.divide(2.0);     // all elements / 2

// Element-wise squaring
Matrix squared = a.square();
// [1,  4]
// [9, 16]

// Sum all elements
double total = a.sum();  // 10.0
```

### Matrix Multiplication

Two distinct multiplication operations are provided — make sure you pick the right one.

```java
double[][] aData = { {1, 3, 2}, {4, 0, 1} };   // 2x3
double[][] bData = { {1, 3}, {0, 1}, {5, 2} };  // 3x2

Matrix a = new Matrix(); a.readData(aData);
Matrix b = new Matrix(); b.readData(bData);

// dot()  — standard matrix product (A × B), dimensions must be compatible
Matrix product = a.dot(b);
// [11, 10]
// [ 9, 14]

// multiply(Matrix)  — element-wise (Hadamard) product, dimensions must match exactly
double[][] cData = { {1, 0}, {0, 1} };
double[][] dData = { {5, 6}, {7, 8} };
Matrix c = new Matrix(); c.readData(cData);
Matrix d = new Matrix(); d.readData(dData);

Matrix hadamard = c.multiply(d);
// [5, 0]
// [0, 8]
```

> **Note:** `dot()` is standard matrix multiplication. `multiply(Matrix)` is element-wise (Hadamard) multiplication. The naming can be counter-intuitive, so read the Javadoc carefully.

### Transpose and Inverse

```java
double[][] data = { {1, 2, 3}, {4, 5, 6} };  // 2x3
Matrix m = new Matrix(); m.readData(data);

// Transpose — returns a new 3x2 matrix
Matrix t = m.transpose();
// [1, 4]
// [2, 5]
// [3, 6]

// Inverse — only valid for square matrices
double[][] sq = { {1, 2}, {3, 4} };
Matrix s = new Matrix(); s.readData(sq);

try {
    Matrix inv = s.inverse();
    inv.print();
    // [-2.0,  1.0]
    // [ 1.5, -0.5]
} catch (NonInvertibleMatrixException e) {
    System.out.println("Matrix cannot be inverted: " + e.getMessage());
}
```

> ⚠️ **See the [Known Bug](#known-bug-matrixinverse-mutates-the-original) section below** before calling `inverse()` in production code.

### Element-wise Mapping

The `map()` method accepts any implementation of the `Function` interface and applies it to every element. This is particularly useful for activation functions in neural networks.

```java
double[][] data = { {-1.0, 0.0}, {0.5, 2.0} };
Matrix m = new Matrix(); m.readData(data);

// Using a lambda (Java 8+)
Matrix relu    = m.map(x -> Math.max(0, x));
// [0.0, 0.0]
// [0.5, 2.0]

Matrix sigmoid = m.map(x -> 1.0 / (1.0 + Math.exp(-x)));
// [0.269, 0.5  ]
// [0.622, 0.880]

// Using an anonymous class (pre-Java 8 style)
Matrix abs = m.map(new Function() {
    public double calculate(double x) {
        return Math.abs(x);
    }
});

// Static variant — equivalent, does not require a Matrix instance
Matrix result = Matrix.map(m, x -> x * x);
```

### Utility Methods

```java
Matrix m = new Matrix(3, 3);

m.randomize();       // fill with random values in [-1, 1]
m.ones();            // fill with 1.0
m.zeros();           // fill with 0.0
m.fill(7.5);         // fill with a specific value

boolean square = m.isSquare();    // true
int rows       = m.getRows();     // 3
int cols       = m.getColumns();  // 3

// Access individual elements
double val = m.getValueAt(1, 2);
m.setValueAt(0, 0, 99.0);

// Retrieve an entire column as a single-column Matrix
Matrix col = m.getColumn(1);

// Convert to a flat 1D array (row-major order)
double[] flat = m.toArray();

// Print to stdout
m.print();

// Compare two matrices element by element
Matrix a = new Matrix(); a.readData(new double[][]{{1,2},{3,4}});
Matrix b = new Matrix(); b.readData(new double[][]{{1,2},{3,4}});
boolean equal = a.equals(b);  // true
```

---

## Vector

`Vector` represents a point or direction in 3D space (with 2D convenience constructors). All instance arithmetic methods mutate and return `this`; static versions return a new `Vector`.

### Creating a Vector

```java
Vector v3 = new Vector(1.0, 2.0, 3.0);  // 3D
Vector v2 = new Vector(1.0, 2.0);       // 2D — z is implicitly 0
Vector v0 = new Vector();               // zero vector

// Random 2D unit vector
Vector rand = Vector.random();

// Unit vector from an angle (radians)
Vector dir = Vector.fromAngle(Math.PI / 4);  // 45°
```

### Arithmetic

```java
Vector a = new Vector(1, 2, 3);
Vector b = new Vector(4, 5, 6);

// Instance methods mutate 'a' and return it
a.add(b);       // a is now (5, 7, 9)
a.sub(b);       // a is now (1, 2, 3) again
a.mult(2.0);    // a is now (2, 4, 6)
a.div(2.0);     // a is now (1, 2, 3) again

// Static methods return a new Vector, leaving a and b unchanged
Vector sum  = Vector.add(a, b);
Vector diff = Vector.sub(a, b);
```

### Geometric Operations

```java
Vector a = new Vector(3, 4, 0);

double magnitude = a.mag();          // 5.0
double magSq     = a.magSq();        // 25.0
Vector unit      = a.copy().normalize(); // (0.6, 0.8, 0.0)

// Limit magnitude to a maximum
Vector vel = new Vector(10, 10, 0);
vel.limit(5.0);  // magnitude capped at 5

// Set to a specific magnitude
vel.setMag(3.0);

// Dot product
Vector b = new Vector(1, 0, 0);
double dot = a.dot(b);  // 3.0

// Cross product
Vector c = new Vector(1, 0, 0);
Vector d = new Vector(0, 1, 0);
Vector cross = c.cross(d);  // (0, 0, 1)

// Euclidean distance between two points
double dist = Vector.dist(a, b);

// Heading (2D angle of the vector, in radians)
Vector heading = new Vector(1, 1, 0);
double angle = heading.heading();  // π/4 ≈ 0.785
```

---

## MathUtils

A collection of static utility methods. Import and call directly — no instantiation needed.

### Clamping and Mapping

```java
// Constrain a value within a range
double clamped = MathUtils.constrain(150.0, 0.0, 100.0);  // 100.0
int    clampedInt = MathUtils.constrain(5, 1, 3);          // 3

// Normalise to [0, 1]
double norm = MathUtils.normalize(75.0, 0.0, 100.0);  // 0.75

// Re-map from one range to another (similar to Arduino/Processing map())
double mapped = MathUtils.map(0.5, 0.0, 1.0, 0.0, 255.0);  // 127.5
```

### Interpolation

```java
// Linear interpolation between two values
double lerped = MathUtils.lerp(0.0, 100.0, 0.25);  // 25.0
double lerped2 = MathUtils.lerp(20.0, 80.0, 0.0);  // 20.0

// Step a value towards a goal by a fixed delta
double current = 0.0;
double goal    = 10.0;
current = MathUtils.approach(goal, current, 3.0);  // 3.0
current = MathUtils.approach(goal, current, 3.0);  // 6.0
current = MathUtils.approach(goal, current, 3.0);  // 9.0
current = MathUtils.approach(goal, current, 3.0);  // 10.0 (clamps at goal)
```

### Random Numbers

```java
double d  = MathUtils.nextDouble(0.0, 1.0);  // random double in [0, 1)
float  f  = (float) MathUtils.nextFloat(0f, 1f);  // random float in [0, 1)
int    n  = MathUtils.nextInt(1, 7);          // random int in [1, 6] — simulates a die
```

### Perlin Noise

The library includes a Java port of Ken Perlin's classic noise algorithm, accessible through `MathUtils`.

```java
// 1D noise — useful for smooth random walks
double n1 = MathUtils.noise(0.5);

// 2D noise — useful for terrain heightmaps
double n2 = MathUtils.noise(x * 0.01, y * 0.01);

// 3D noise — useful for animated 2D noise (use time as z)
double n3 = MathUtils.noise(x * 0.01, y * 0.01, frameCount * 0.005);
```

Perlin noise returns values roughly in the range [-1, 1]. Use `MathUtils.map()` to rescale to whatever range you need:

```java
// Rescale noise output to [0, 255] for a greyscale pixel value
double noiseVal = MathUtils.noise(x * 0.01, y * 0.01);
double pixel    = MathUtils.map(noiseVal, -1.0, 1.0, 0.0, 255.0);
```

---

## Function Interface

`Function` is a single-method interface used by `Matrix.map()`. In Java 8 and later it can be satisfied with a lambda expression; in earlier versions, use an anonymous class.

```java
// As a lambda
Function sigmoid = x -> 1.0 / (1.0 + Math.exp(-x));
Function relu    = x -> Math.max(0, x);
Function tanh    = x -> Math.tanh(x);

// As a named class
public class Sigmoid implements Function {
    @Override
    public double calculate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
```

Apply to a matrix:

```java
Matrix activations = weightedSums.map(sigmoid);
```

---

## Known Bug: `Matrix.inverse()` Mutates the Original

There is a subtle but significant bug in the `inverse()` method. It calls the private helper `gaussian()`, passing the internal `matrix` array directly:

```java
// Inside inverse():
gaussian(matrix, index);   // ← 'matrix' is the raw internal array
```

The `gaussian()` method performs Gaussian elimination **in place**, modifying the values of the array it receives. Because Java passes arrays by reference, this means `gaussian()` overwrites the contents of the original matrix rather than working on a copy. After calling `inverse()`, the original matrix is left in a corrupted, partially-eliminated state.

**Example of the problem:**

```java
double[][] data = { {1, 2}, {3, 4} };
Matrix m = new Matrix();
m.readData(data);

System.out.println("Before:");
m.print();
// [1.0, 2.0]
// [3.0, 4.0]

Matrix inv = m.inverse();

System.out.println("After calling inverse(), m is now:");
m.print();
// ← m is corrupted; its values have been overwritten by the elimination process
```

**The fix** is straightforward — pass a deep copy of the internal array to `gaussian()` instead of the original:

```java
public Matrix inverse() throws NonInvertibleMatrixException {
    if (!isSquare())
        throw new NonInvertibleMatrixException("Cannot invert a non square matrix.");

    int n = rows;
    double x[][] = new double[n][n];
    double b[][] = new double[n][n];
    int index[]  = new int[n];

    // Take a deep copy so the original matrix is not mutated
    double[][] copy = new double[n][n];
    for (int i = 0; i < n; i++)
        copy[i] = matrix[i].clone();

    for (int i = 0; i < n; i++)
        b[i][i] = 1;

    gaussian(copy, index);   // ← pass 'copy', not 'matrix'

    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            for (int k = 0; k < n; k++)
                b[index[j]][k] -= copy[index[j]][i] * b[index[i]][k];

    for (int i = 0; i < n; i++) {
        x[n-1][i] = b[index[n-1]][i] / copy[index[n-1]][n-1];
        for (int j = n - 2; j >= 0; j--) {
            x[j][i] = b[index[j]][i];
            for (int k = j + 1; k < n; k++)
                x[j][i] -= copy[index[j]][k] * x[k][i];
            x[j][i] /= copy[index[j]][j];
        }
    }

    Matrix result = new Matrix();
    result.readData(x);
    return result;
}
```

This bug is easy to miss because the returned inverse matrix itself is computed correctly — it is only the **caller's copy of the original** that is silently destroyed. In short-lived or single-use code this may go unnoticed, but in any iterative algorithm (such as a training loop) that repeatedly inverts and then uses the same matrix, the results will be wrong from the second call onwards.
