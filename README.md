

# JavaMathLib

A lightweight Java mathematics library providing matrix algebra, vector maths, Perlin noise generation, and a collection of general-purpose numeric utilities. The library is designed with machine learning applications in mind, but is broadly useful for simulations, procedural graphics, and any numerically intensive Java project.

---

## Table of Contents

- [Classes at a Glance](#classes-at-a-glance)
- [Getting Started](#getting-started)
- [Matrix](#matrix)
  - [Creating a Matrix](#creating-a-matrix)
  - [Arithmetic Operations](#arithmetic-operations)
  - [Matrix Multiplication](#matrix-multiplication)
  - [Transpose and Inverse](#transpose-and-inverse)
  - [Solving Linear Systems](#solving-linear-systems)
  - [LU Decomposition](#lu-decomposition)
  - [Determinant, Trace and Mean](#determinant-trace-and-mean)
  - [Element-wise Mapping](#element-wise-mapping)
  - [Reshaping and Combining](#reshaping-and-combining)
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
- [Running the Tests](#running-the-tests)
- [Changelog](#changelog)

---

## Classes at a Glance

| Class | Description |
|---|---|
| `Matrix` | Full-featured 2D matrix with ML-oriented operations |
| `Matrix.LU` | Reusable LU factorisation of a square matrix (`P·A = L·U`) |
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

No external dependencies are required. Java 8 or later is needed.

---

## Matrix

The `Matrix` class wraps a `double[][]` array and exposes a rich set of operations suitable for neural networks and other ML workloads.

Every arithmetic and structural operation (`add`, `dot`, `transpose`, `inverse`, `solve`, `map`, `appendRight` and the rest) returns a **new** `Matrix` and never modifies the one it is called on. The only methods that change a matrix in place are `setValueAt`, `readData`, `randomize`, `ones`, `zeros` and `fill`.

### Creating a Matrix

```java
// Directly from a 2D array (the data is copied)
Matrix m = new Matrix(new double[][] { {1, 2, 3}, {4, 5, 6} });

// Empty matrix — populate later with readData()
Matrix a = new Matrix();
a.readData(new double[][] { {1, 2, 3}, {4, 5, 6} });

// Pre-sized matrix, all elements initialised to 0
Matrix b = new Matrix(3, 3);

// n x n identity matrix
Matrix i3 = Matrix.identity(3);

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

// Inverse — only valid for square, non-singular matrices
Matrix s = new Matrix(new double[][] { {1, 2}, {3, 4} });

try {
    Matrix inv = s.inverse();   // s itself is left unchanged
    inv.print();
    // [-2.0,  1.0]
    // [ 1.5, -0.5]
} catch (NonInvertibleMatrixException e) {
    System.out.println("Matrix cannot be inverted: " + e.getMessage());
}
```

`inverse()` throws `NonInvertibleMatrixException` if the matrix is not square, is empty, or is singular (or so close to singular that the result would be meaningless — see `Matrix.SINGULAR_TOLERANCE`). Previously a singular matrix silently "inverted" to a matrix of `Infinity` values.

### Solving Linear Systems

To solve `A·x = b`, use `solve()` rather than `inverse().dot(b)` — it does the same elimination without building the full inverse, so it is faster and more accurate.

```java
Matrix A = new Matrix(new double[][] { {2, 1, -1}, {-3, -1, 2}, {-2, 1, 2} });
Matrix b = Matrix.fromArray(new double[] { 8, -11, -3 });

Matrix x = A.solve(b);
// 2.0
// 3.0
// -1.0
```

`b` may have several columns; each is solved as a separate right-hand side. (`A.solve(Matrix.identity(n))` is exactly `A.inverse()`.)

Each call to `solve()` factorises `A` from scratch. If you need to solve against the same `A` many times, factorise it once with `lu()` — see below.

### LU Decomposition

`lu()` factorises a square matrix into `P·A = L·U`, where `P` is a permutation matrix (the row swaps made while pivoting), `L` is lower triangular with ones on its diagonal, and `U` is upper triangular. `solve()`, `inverse()` and `determinant()` are all built on it.

The reason to call it yourself is reuse. Factorising costs about n³ operations; each solve with an existing factorisation costs only about n². So when the matrix stays the same but the right-hand side changes — Newton iterations, a Kalman filter, a simulation stepping through time, regression against several targets — factorise once and keep the result:

```java
Matrix A = new Matrix(new double[][] { {2, 1, -1}, {-3, -1, 2}, {-2, 1, 2} });

try {
    Matrix.LU lu = A.lu();                 // the expensive part, done once

    Matrix x1 = lu.solve(b1);              // each of these is cheap
    Matrix x2 = lu.solve(b2);
    double det = lu.determinant();         // -1.0, essentially free
    Matrix inv = lu.inverse();

    // Inspect the factors (each getter returns a copy)
    Matrix L = lu.getL();
    Matrix U = lu.getU();
    Matrix P = lu.getP();
    boolean ok = P.dot(A).equals(L.dot(U), 1e-12);  // true
    int[] pivot = lu.getPivot();           // compact form of P: row i came from A's row pivot[i]
} catch (NonInvertibleMatrixException e) {
    System.out.println("Matrix is singular: " + e.getMessage());
}
```

On a 400×400 system, factorising once and solving 50 right-hand sides takes roughly 40 ms, against roughly 280 ms for 50 separate `solve()` calls.

A `Matrix.LU` is immutable, so it is safe to keep and share. `lu()` throws `NonInvertibleMatrixException` if the matrix is not square, is empty, or is singular. Pivoting is scaled (each candidate pivot is compared relative to the size of its row), so a matrix that is merely small, such as `identity(3).multiply(1e-20)`, is not mistaken for a singular one.

### Determinant, Trace and Mean

```java
Matrix s = new Matrix(new double[][] { {1, 2}, {3, 4} });

double det   = s.determinant();  // -2.0
double trace = s.trace();        //  5.0  (sum of the diagonal)
double mean  = s.mean();         //  2.5  (average of all elements)

// Mean squared error in one line
double mse = prediction.subtract(target).square().mean();
```

`determinant()` uses the same LU factorisation and the same singularity test as `inverse()`, so the two always agree: for a matrix that `inverse()` rejects, such as `{{1,2,3},{4,5,6},{7,8,9}}`, `determinant()` returns exactly `0` rather than a rounding-error value like `6.66e-16`.

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

### Reshaping and Combining

```java
Matrix m = new Matrix(new double[][] { {1, 2}, {3, 4} });

// Side by side (row counts must match) — e.g. add a bias column of ones
Matrix ones = new Matrix(2, 1); ones.ones();
Matrix withBias = m.appendRight(ones);
// [1, 2, 1]
// [3, 4, 1]

// Stacked (column counts must match)
Matrix stacked = m.appendBottom(new Matrix(new double[][] { {5, 6} }));
// [1, 2]
// [3, 4]
// [5, 6]

// Pad with zeros
Matrix taller = m.addRows(1);     // 3x2, last row zeros
Matrix wider  = m.addColumns(2);  // 2x4, last two columns zeros

// Single rows and columns
Matrix row = m.getRow(0);         // [1, 2]  (1x2)
Matrix col = m.getColumn(1);      // [2, 4]  (2x1)

// Flatten (row-major) into a column or row vector
Matrix asCol = m.getAsVector(Matrix.COLUMN_VECTOR);  // 4x1
Matrix asRow = m.getAsVector(Matrix.ROW_VECTOR);     // 1x4
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

// Print to stdout (columns are right-aligned)
m.print();

// Compare two matrices element by element
Matrix a = new Matrix(new double[][]{{1,2},{3,4}});
Matrix b = new Matrix(new double[][]{{1,2},{3,4}});
boolean equal = a.equals(b);  // true

// Compare within a tolerance — usually what you want after floating point maths
boolean isIdentity = a.dot(a.inverse()).equals(Matrix.identity(2), 1e-9);  // true
```

`Matrix` also overrides `equals(Object)` and `hashCode()`, so matrices behave correctly as keys in a `HashMap` or members of a `HashSet`.

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
float  f  = MathUtils.nextFloat(0f, 1f);     // random float in [0, 1)
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

## Running the Tests

`MatrixTest.java` is a self-contained test suite with no dependencies (no JUnit needed):

```bash
javac *.java
java MatrixTest
# 162 passed, 0 failed
```

---

## Changelog

### LU decomposition release

#### New features

- **`lu()` and the `Matrix.LU` class.** Factorise a square matrix once as `P·A = L·U`, then reuse it: `solve(b)` (about n² per right-hand side instead of n³), `inverse()`, `determinant()`, plus `getL()`, `getU()`, `getP()`, `getPivot()` and `size()`.
- `solve()`, `inverse()` and `determinant()` are now all built on this single factorisation, replacing two separate elimination routines. Their signatures and exceptions are unchanged, so existing code needs no changes.

#### Improvements

- **`determinant()` now agrees with `inverse()` about singularity.** It previously ran its own elimination with unscaled pivoting and an exact `== 0` test, so a matrix that `inverse()` correctly rejected could have a determinant of `6.66e-16`. It now returns exactly `0` for any matrix `inverse()` would reject.
- `inverse()` is about twice as fast: the factorisation stores its rows already swapped, which avoids indirect indexing in the inner loops.
- Test suite extended from 74 to 162 checks, including `P·A = L·U`, triangularity of `L` and `U`, reuse of one factorisation across many right-hand sides, and scale invariance of the singularity test.
- README: the `nextFloat` example no longer needs a `(float)` cast, and the test count is corrected.

### Earlier release

#### Bug fixes

- **`inverse()` no longer destroys the original matrix.** It used to run Gaussian elimination directly on the matrix's own internal array, so after one call the matrix was left half-eliminated and every later calculation with it — including a second `inverse()` — was wrong. Elimination now always works on a private copy.
- **Singular matrices now throw `NonInvertibleMatrixException`** instead of silently returning `Infinity`/`NaN` values. A latent row-swap bug in the pivot search (the pivot index carried over between columns) is fixed at the same time.
- `getValueAt()` and `getColumn()` accepted a column index equal to the column count; they now reject it with a clear message.
- `readData()` rejected empty arrays with an `ArrayIndexOutOfBoundsException` and silently accepted jagged arrays; it now accepts empty data and rejects jagged rows. `new Matrix().copy()` no longer throws a `NullPointerException`.
- `dot()` no longer crashes when the right-hand matrix has zero rows.
- `equals(Matrix)` now handles `null`; `equals(Object)` and a matching `hashCode()` have been added (with `0.0` and `-0.0` hashing alike, since they compare equal).
- Javadoc corrections: the class example used `multiply` where it meant `dot`, and `toArray()` was described as column-ordered when it is row-major.

#### New features

- `new Matrix(double[][])` constructor
- `Matrix.identity(n)` and `identity()` (previously unimplemented)
- `solve(b)` for linear systems
- `determinant()`, `trace()`, `mean()`
- `getRow()`, `appendRight()`, `appendBottom()`, `addRows()`, `addColumns()` (previously unimplemented — now return new matrices rather than changing the original)
- `getAsVector(COLUMN_VECTOR | ROW_VECTOR)` — the constants existed but the method did not
- `equals(Matrix, tolerance)`
- `toString()` right-aligns columns
- The demo `main()` has been removed from `Matrix` and replaced by `MatrixTest`
