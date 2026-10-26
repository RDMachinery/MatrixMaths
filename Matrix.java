import java.util.function.DoubleBinaryOperator;

/**
 * A special purpose matrix maths class built for implementing machine
 * learning algorithms. This class may be suitable
 * for programs that make use of matrix computations and matrix
 * manipulations other than machine learning projects.
 * <p>
 * All arithmetic and structural operations (<code>add</code>, <code>subtract</code>,
 * <code>multiply</code>, <code>divide</code>, <code>dot</code>, <code>transpose</code>,
 * <code>inverse</code>, <code>solve</code>, <code>map</code>, <code>appendRight</code> and so on)
 * return a <b>new</b> matrix and never modify the matrix they are called on. The only
 * methods that change a matrix in place are the explicit setters and fillers:
 * <code>setValueAt</code>, <code>readData</code>, <code>randomize</code>, <code>ones</code>,
 * <code>zeros</code> and <code>fill</code>.
 * <p>
 * The example below creates two matrices and then multiplies them
 * together:
 * <pre>
 * Matrix a = new Matrix(new double[][] {{1,3,2},{4,0,1}});
 * Matrix b = new Matrix(new double[][] {{1,3},{0,1},{5,2}});
 * Matrix c = a.dot(b);      // standard matrix product
 * Matrix d = a.multiply(a); // element-wise (Hadamard) product
 * </pre>
 *
 * @author Mario Gianota 29 June 2019 gianotamario@gmail.com
 */
public class Matrix {
    double[][] matrix;
    int rows;
    int columns;

    /**
     * Used as an argument to the <code>getAsVector</code> method to
     * return a column vector.
     * @see #getAsVector(int)
     */
    public static final int COLUMN_VECTOR = 0;
    /**
     * Used as an argument to the <code>getAsVector</code> method to
     * return a row vector.
     * @see #getAsVector(int)
     */
    public static final int ROW_VECTOR = 1;

    /**
     * Pivots smaller than this (relative to the size of their row) are treated as
     * zero when inverting or solving, so that singular and numerically singular
     * matrices are reported rather than producing Infinity or NaN.
     */
    public static final double SINGULAR_TOLERANCE = 1e-12;

    /**
     * Creates a new 0x0 matrix. Use the <code>readData</code> method to
     * load data into the matrix.
     * @see #readData
     */
    public Matrix() {
        matrix = new double[0][0];
    }

    /**
     * Creates a new matrix with the specified number of rows and
     * columns. All of the elements are initialised to zero.
     *
     * @param numRows Number of rows
     * @param numCols Number of columns
     */
    public Matrix(int numRows, int numCols) {
        if( numRows < 0 || numCols < 0)
            throw new IllegalArgumentException("Negative dimension: " + numRows + "x" + numCols);
        matrix = new double[numRows][numCols];
        rows = numRows;
        columns = numCols;
    }

    /**
     * Creates a new matrix holding a copy of the supplied data. Equivalent to
     * constructing an empty matrix and calling <code>readData(data)</code>.
     *
     * @param data The data to copy into the matrix
     */
    public Matrix(double[][] data) {
        load(data);
    }

    // ------------------------------------------------------------------
    // Construction helpers
    // ------------------------------------------------------------------

    /** Wraps a freshly created array without copying it. */
    private static Matrix adopt(double[][] data, int cols) {
        Matrix m = new Matrix();
        m.matrix = data;
        m.rows = data.length;
        m.columns = cols;
        return m;
    }

    private static double[][] deepCopy(double[][] src) {
        double[][] dst = new double[src.length][];
        for(int i=0; i<src.length; i++)
            dst[i] = src[i].clone();
        return dst;
    }

    /**
     * Returns an <code>n</code> x <code>n</code> identity matrix, in which the
     * diagonal is all ones and all other elements are zero.
     *
     * @param n The size of the matrix
     * @return Identity matrix
     */
    public static Matrix identity(int n) {
        Matrix m = new Matrix(n, n);
        for(int i=0; i<n; i++)
            m.matrix[i][i] = 1;
        return m;
    }

    /**
     * Returns an identity matrix with the same dimensions as this matrix. This
     * matrix must be square.
     *
     * @return Identity matrix
     * @see #identity(int)
     */
    public Matrix identity() {
        requireSquare("identity");
        return identity(rows);
    }

    /**
     * Converts an array to a matrix. The returned matrix has all elements
     * of the supplied array in a single column.
     *
     * @param vector The array
     * @return the Matrix
     */
    public static Matrix fromArray(double[] vector) {
        Matrix m = new Matrix(vector.length,1);
        for(int row=0; row<vector.length; row++)
            m.matrix[row][0] = vector[row];
        return m;
    }

    // ------------------------------------------------------------------
    // Multiplication
    // ------------------------------------------------------------------

    /**
     * Computes the matrix product of this matrix by the supplied matrix <code>m</code> and
     * returns the result. This matrix is left unchanged. If A is this matrix and B is the supplied
     * matrix <code>m</code>, then this method performs the following
     * matrix calculation: <code>A*B</code>
     *
     * @param m The matrix to multiply by
     * @return result of multiplying this matrix by m
     * @see #multiply(Matrix)
     */
    public Matrix dot(Matrix m) {
        if( columns != m.rows )
            throw new IllegalArgumentException("Can't multiply matrices. "+
                    "Number of columns in left hand side do not match number of rows in right handside: "+
                    columns + "!=" + m.rows);

        double[][] rhs = m.matrix;
        double[][] result = new double[rows][m.columns];
        // i-k-j loop order walks both arrays row by row, which is cache friendly
        for(int i=0; i<rows; i++) {
            double[] resultRow = result[i];
            for(int k=0; k<columns; k++) {
                double aik = matrix[i][k];
                double[] rhsRow = rhs[k];
                for(int j=0; j<m.columns; j++)
                    resultRow[j] += aik * rhsRow[j];
            }
        }
        return adopt(result, m.columns);
    }

    /**
     * Multiplies this matrix element by element with the supplied matrix <code>m</code> (the
     * Hadamard product) and returns the result. This matrix remains unchanged. The number of
     * rows and columns in both matrices must agree for this operation to succeed.
     *
     * @param m The matrix to multiply by
     * @return result of multiplying this matrix by m
     * @see #dot(Matrix)
     */
    public Matrix multiply(Matrix m) {
        return elementwise(m, "element multiply", (a, b) -> a * b);
    }

    /**
     * Multiplies every component of this matrix by the value <code>n</code> and returns the
     * result.
     *
     * @param n Multiplier
     * @return the result
     */
    public Matrix multiply(double n) {
        return map(x -> x * n);
    }

    /**
     * Divides every number in the matrix by the value <code>n</code> and returns the
     * result.
     *
     * @param n Divisor
     * @return Division result
     */
    public Matrix divide(double n) {
        if( n == 0 )
            throw new IllegalArgumentException("Cannot divide matrix by zero.");
        return map(x -> x / n);
    }

    /**
     * Computes the square of this matrix, that is the matrix with
     * all numbers squared, and returns the result. (This is element-wise squaring,
     * not <code>A*A</code>; use <code>dot</code> for that.)
     *
     * @return The result
     */
    public Matrix square() {
        return map(x -> x * x);
    }

    // ------------------------------------------------------------------
    // Addition and subtraction
    // ------------------------------------------------------------------

    /**
     * Adds a matrix to this matrix and returns the result. The
     * supplied matrix must be of the same dimension for this
     * operation to succeed.
     *
     * @param m The matrix to add
     * @return the result
     */
    public Matrix add(Matrix m) {
        return elementwise(m, "add", (a, b) -> a + b);
    }

    /**
     * Adds <code>n</code> to every number in the matrix and
     * returns the result.
     *
     * @param n The number to add
     * @return The result
     */
    public Matrix add(double n) {
        return map(x -> x + n);
    }

    /**
     * Subtracts the supplied matrix <code>m</code> from the matrix. If the
     * dimensions of the supplied matrix do not match then an <code>IllegalArgumentException</code>
     * is thrown.
     *
     * @param m The matrix to subtract
     * @return The result
     */
    public Matrix subtract(Matrix m) {
        return elementwise(m, "subtract", (a, b) -> a - b);
    }

    /**
     * Subtracts <code>n</code> from every number in the matrix and
     * returns the result.
     *
     * @param n The number to subtract
     * @return The result
     */
    public Matrix subtract(double n) {
        return map(x -> x - n);
    }

    private Matrix elementwise(Matrix m, String opName, DoubleBinaryOperator op) {
        if( m.rows != rows || m.columns != columns ) {
            throw new IllegalArgumentException("Cannot " + opName + " matrices. "+
                    "Number of rows and columns do not match: "+
                    rows + "x" + columns + " != " + m.rows + "x" + m.columns);
        }
        double[][] result = new double[rows][columns];
        for(int i=0; i<rows; i++)
            for(int j=0; j<columns; j++)
                result[i][j] = op.applyAsDouble(matrix[i][j], m.matrix[i][j]);
        return adopt(result, columns);
    }

    // ------------------------------------------------------------------
    // Reductions
    // ------------------------------------------------------------------

    /**
     * Sums all numbers in the matrix and returns the result.
     *
     * @return the sum
     */
    public double sum() {
        double sum = 0d;
        for(int i=0; i<rows; i++)
            for(int j=0; j<columns; j++)
                sum += matrix[i][j];
        return sum;
    }

    /**
     * Returns the mean (average) of all numbers in the matrix. Useful, for example,
     * for computing mean squared error: <code>prediction.subtract(target).square().mean()</code>.
     *
     * @return the mean
     */
    public double mean() {
        if( rows == 0 || columns == 0 )
            throw new IllegalStateException("Cannot take the mean of an empty matrix.");
        return sum() / ((double) rows * columns);
    }

    /**
     * Returns the trace of the matrix, the sum of the elements on the main diagonal.
     * The matrix must be square.
     *
     * @return the trace
     */
    public double trace() {
        requireSquare("trace");
        double t = 0;
        for(int i=0; i<rows; i++)
            t += matrix[i][i];
        return t;
    }

    /**
     * Computes the determinant of the matrix. The matrix must be square and is left
     * unchanged.
     * <p>
     * The determinant is taken from the same LU factorisation used by <code>solve</code>
     * and <code>inverse</code>, so the three always agree: if <code>inverse()</code> would
     * report the matrix as singular, this method returns exactly <code>0</code> rather than
     * a tiny rounding-error value such as <code>6.66e-16</code>.
     *
     * @return the determinant
     * @see #lu()
     */
    public double determinant() {
        requireSquare("determinant");
        if( rows == 0 )
            return 1; // by convention, the determinant of a 0x0 matrix is 1
        LU f = new LU(matrix, rows);
        return f.singularColumn >= 0 ? 0 : f.det();
    }

    // ------------------------------------------------------------------
    // Transpose, LU factorisation, inverse and solving
    // ------------------------------------------------------------------

    /**
     * Returns the matrix's transpose.
     *
     * @return the transposed matrix
     */
    public Matrix transpose() {
        double[][] ret = new double[columns][rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                ret[j][i] = matrix[i][j];
        return adopt(ret, rows);
    }

    /**
     * Computes the LU factorisation of this matrix, <code>P*A = L*U</code>, using Gaussian
     * elimination with scaled partial pivoting. This matrix is left unchanged.
     * <p>
     * Factorising costs about n<sup>3</sup> operations, but each later
     * <code>solve</code> on the result costs only about n<sup>2</sup>. When you need to solve
     * against the same matrix many times, factorise once and reuse the result:
     * <pre>
     * Matrix.LU lu = A.lu();
     * Matrix x1 = lu.solve(b1);
     * Matrix x2 = lu.solve(b2);
     * double det = lu.determinant();
     * </pre>
     *
     * @return The factorisation
     * @exception NonInvertibleMatrixException If the matrix is not square, is empty or is
     * singular (or numerically singular; see {@link #SINGULAR_TOLERANCE})
     * @see LU
     */
    public LU lu() throws NonInvertibleMatrixException {
        if( ! isSquare() )
            throw new NonInvertibleMatrixException("Cannot factorise a non square matrix: " + rows + "x" + columns);
        if( rows == 0 )
            throw new NonInvertibleMatrixException("Cannot factorise an empty matrix.");
        LU f = new LU(matrix, rows);
        f.requireNonSingular();
        return f;
    }

    /**
     * Computes the inverse of the matrix. This matrix is left unchanged.
     * <p>
     * If you only need the inverse in order to compute <code>inverse().dot(b)</code>,
     * call <code>solve(b)</code> instead: it is faster and more accurate.
     *
     * @return The inverse
     * @exception NonInvertibleMatrixException If the matrix is not square, is empty,
     * or is singular (or so close to singular that its inverse would be meaningless)
     * @see #solve(Matrix)
     * @see #lu()
     */
    public Matrix inverse() throws NonInvertibleMatrixException {
        if( ! isSquare() )
            throw new NonInvertibleMatrixException("Cannot invert a non square matrix: " + rows + "x" + columns);
        return lu().inverse();
    }

    /**
     * Solves the linear system <code>A*X = B</code> for <code>X</code>, where A is this
     * (square) matrix and B is the supplied matrix <code>rhs</code>. B may have any number of
     * columns; each column is solved as a separate right hand side. Neither matrix is
     * modified.
     * <p>
     * Each call factorises A afresh. To solve against the same A repeatedly, call
     * <code>lu()</code> once and use {@link LU#solve(Matrix)}.
     *
     * @param rhs The right hand side B. It must have the same number of rows as this matrix.
     * @return X, a matrix with the same dimensions as <code>rhs</code>
     * @exception NonInvertibleMatrixException If this matrix is not square, is empty or is singular
     * @see #lu()
     */
    public Matrix solve(Matrix rhs) throws NonInvertibleMatrixException {
        if( ! isSquare() )
            throw new NonInvertibleMatrixException("Cannot solve with a non square matrix: " + rows + "x" + columns);
        if( rows == 0 )
            throw new NonInvertibleMatrixException("Cannot solve with an empty matrix.");
        if( rhs.rows != rows )
            throw new IllegalArgumentException("Right hand side must have " + rows +
                    " rows but has " + rhs.rows);
        return lu().solve(rhs);
    }

    /**
     * The LU factorisation of a square matrix A, in the form <code>P*A = L*U</code>, where
     * P is a permutation matrix, L is unit lower triangular (ones on the diagonal) and U is
     * upper triangular. Obtain one with {@link Matrix#lu()}.
     * <p>
     * An <code>LU</code> is immutable: none of its methods change it, and the matrices
     * returned by <code>getL</code>, <code>getU</code> and <code>getP</code> are copies.
     * It is therefore safe to keep one and reuse it for as many solves as you like.
     */
    public static final class LU {
        /** L (below the diagonal, unit diagonal implied) and U (on and above), rows already permuted. */
        private final double[][] lu;
        /** perm[i] is the row of the original matrix that ended up in row i. */
        private final int[] perm;
        private final int n;
        /** +1 or -1: the sign of the permutation (an even or odd number of row swaps). */
        private final int sign;
        /** -1 if the matrix is non singular, otherwise the column where no usable pivot was found. */
        private final int singularColumn;
        /** -1, or the index of an all-zero row (reported in preference to singularColumn). */
        private final int zeroRow;

        /**
         * Factorises a copy of <code>a</code>. Never throws for singularity: it records
         * where elimination stopped, so that <code>determinant()</code> can return 0 while
         * <code>lu()</code> reports the problem.
         */
        private LU(double[][] a, int n) {
            this.n = n;
            double[][] w = deepCopy(a);
            int[] p = new int[n];
            double[] scale = new double[n];
            int s = 1, badCol = -1, badRow = -1;

            for (int i = 0; i < n; i++) {
                p[i] = i;
                double max = 0;
                for (int j = 0; j < n; j++)
                    max = Math.max(max, Math.abs(w[i][j]));
                if (max == 0 && badRow < 0)
                    badRow = i;
                scale[i] = max;
            }

            if (badRow < 0) {
                for (int j = 0; j < n; j++) {
                    // Choose the pivot that is largest relative to the size of its row
                    int k = j;
                    double best = 0;
                    for (int i = j; i < n; i++) {
                        double r = Math.abs(w[i][j]) / scale[i];
                        if (r > best) { best = r; k = i; }
                    }
                    if (best < SINGULAR_TOLERANCE) { badCol = j; break; }

                    if (k != j) {
                        double[] t = w[k]; w[k] = w[j]; w[j] = t;
                        int ti = p[k]; p[k] = p[j]; p[j] = ti;
                        double ts = scale[k]; scale[k] = scale[j]; scale[j] = ts;
                        s = -s;
                    }

                    double pivot = w[j][j];
                    double[] pivotRow = w[j];
                    for (int i = j + 1; i < n; i++) {
                        double[] row = w[i];
                        double f = row[j] / pivot;
                        row[j] = f;                       // store the multiplier: this is L
                        if (f != 0)
                            for (int l = j + 1; l < n; l++)
                                row[l] -= f * pivotRow[l];
                    }
                }
            }
            this.lu = w;
            this.perm = p;
            this.sign = s;
            this.singularColumn = badRow >= 0 ? 0 : badCol;
            this.zeroRow = badRow;
        }

        private void requireNonSingular() throws NonInvertibleMatrixException {
            if (zeroRow >= 0)
                throw new NonInvertibleMatrixException("Matrix is singular: row " + zeroRow + " is all zeros.");
            if (singularColumn >= 0)
                throw new NonInvertibleMatrixException("Matrix is singular (no usable pivot in column " + singularColumn + ").");
        }

        private double det() {
            double d = sign;
            for (int i = 0; i < n; i++)
                d *= lu[i][i];
            return d;
        }

        /**
         * Returns the size n of the n x n matrix that was factorised.
         * @return n
         */
        public int size() {
            return n;
        }

        /**
         * Solves <code>A*X = B</code> for X using this factorisation. B may have any number
         * of columns. Costs about n<sup>2</sup> operations per column of B. B is not modified.
         *
         * @param rhs The right hand side B, with <code>size()</code> rows
         * @return X, a matrix with the same dimensions as <code>rhs</code>
         */
        public Matrix solve(Matrix rhs) {
            if (rhs.rows != n)
                throw new IllegalArgumentException("Right hand side must have " + n +
                        " rows but has " + rhs.rows);
            int m = rhs.columns;
            double[][] x = new double[n][];

            // Apply the permutation:  y = P*B
            for (int i = 0; i < n; i++)
                x[i] = rhs.matrix[perm[i]].clone();

            // Forward substitution:  L*y = P*B   (L has a unit diagonal)
            for (int i = 1; i < n; i++) {
                double[] xi = x[i], li = lu[i];
                for (int k = 0; k < i; k++) {
                    double f = li[k];
                    if (f != 0) {
                        double[] xk = x[k];
                        for (int c = 0; c < m; c++)
                            xi[c] -= f * xk[c];
                    }
                }
            }

            // Back substitution:  U*x = y
            for (int i = n - 1; i >= 0; i--) {
                double[] xi = x[i], ui = lu[i];
                for (int k = i + 1; k < n; k++) {
                    double f = ui[k];
                    if (f != 0) {
                        double[] xk = x[k];
                        for (int c = 0; c < m; c++)
                            xi[c] -= f * xk[c];
                    }
                }
                double d = ui[i];
                for (int c = 0; c < m; c++)
                    xi[c] /= d;
            }
            return adopt(x, m);
        }

        /**
         * Returns the inverse of the factorised matrix.
         * @return the inverse
         */
        public Matrix inverse() {
            return solve(identity(n));
        }

        /**
         * Returns the determinant of the factorised matrix: the product of U's diagonal,
         * negated if an odd number of row swaps was made. Costs about n operations.
         * @return the determinant
         */
        public double determinant() {
            return det();
        }

        /**
         * Returns L, the unit lower triangular factor (ones on the diagonal).
         * @return a copy of L
         */
        public Matrix getL() {
            double[][] l = new double[n][n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(lu[i], 0, l[i], 0, i);
                l[i][i] = 1;
            }
            return adopt(l, n);
        }

        /**
         * Returns U, the upper triangular factor.
         * @return a copy of U
         */
        public Matrix getU() {
            double[][] u = new double[n][n];
            for (int i = 0; i < n; i++)
                System.arraycopy(lu[i], i, u[i], i, n - i);
            return adopt(u, n);
        }

        /**
         * Returns P, the permutation matrix such that <code>P*A = L*U</code>.
         * @return P
         */
        public Matrix getP() {
            Matrix p = new Matrix(n, n);
            for (int i = 0; i < n; i++)
                p.matrix[i][perm[i]] = 1;
            return p;
        }

        /**
         * Returns the pivot order as an array: element i is the row of the original
         * matrix that became row i. This is a compact alternative to <code>getP()</code>.
         * @return a copy of the pivot order
         */
        public int[] getPivot() {
            return perm.clone();
        }
    }

    // ------------------------------------------------------------------
    // Mapping
    // ------------------------------------------------------------------

    /**
     * Apply a function to every element of the matrix and return
     * the result. This matrix is left unchanged.
     *
     * @param f The function to call
     * @return The result of applying the function
     * @see Function
     */
    public Matrix map(Function f) {
        double[][] result = new double[rows][columns];
        for(int i=0; i<rows; i++)
            for(int j=0; j<columns; j++)
                result[i][j] = f.calculate(matrix[i][j]);
        return adopt(result, columns);
    }

    /**
     * Static method to apply a function to each element of the supplied matrix and
     * return the result. The supplied matrix is left unchanged.
     *
     * @param matrix The matrix
     * @param function The function to apply
     * @return The result of applying the function
     * @see Function
     */
    public static Matrix map(Matrix matrix, Function function) {
        return matrix.map(function);
    }

    // ------------------------------------------------------------------
    // Structure
    // ------------------------------------------------------------------

    /**
     * Returns the row of values located at the supplied row index as a
     * one row Matrix.
     *
     * @param row The row index
     * @return The row as a one row Matrix
     */
    public Matrix getRow(int row) {
        if( row < 0 || row >= rows )
            throw new IllegalArgumentException("No such row: " + row);
        return adopt(new double[][] { matrix[row].clone() }, columns);
    }

    /**
     * Returns the column of values located at the supplied column index. The returned
     * Matrix is a one column Matrix with the number of rows equal to the number of
     * rows in this Matrix.
     *
     * @param column The column index
     * @return The column as a one column Matrix
     */
    public Matrix getColumn(int column) {
        if( column < 0 || column >= columns )
            throw new IllegalArgumentException("No such column: " + column);
        Matrix m = new Matrix(rows,1);
        for(int i=0; i<rows; i++)
            m.matrix[i][0] = matrix[i][column];
        return m;
    }

    /**
     * Returns all of the elements of this matrix, in row-major order, as either a single
     * column matrix (<code>COLUMN_VECTOR</code>) or a single row matrix (<code>ROW_VECTOR</code>).
     *
     * @param type <code>COLUMN_VECTOR</code> or <code>ROW_VECTOR</code>
     * @return The flattened matrix
     */
    public Matrix getAsVector(int type) {
        double[] flat = toArray();
        if( type == COLUMN_VECTOR )
            return fromArray(flat);
        if( type == ROW_VECTOR )
            return adopt(new double[][] { flat }, flat.length);
        throw new IllegalArgumentException("Unknown vector type: " + type);
    }

    /**
     * Returns a new matrix that is this matrix with <code>numRows</code> rows of zeros
     * added to the bottom. This matrix is left unchanged.
     *
     * @param numRows Number of rows to add
     * @return The enlarged matrix
     */
    public Matrix addRows(int numRows) {
        if( numRows < 0 )
            throw new IllegalArgumentException("Negative number of rows: " + numRows);
        return appendBottom(new Matrix(numRows, columns));
    }

    /**
     * Returns a new matrix that is this matrix with <code>numColumns</code> columns of zeros
     * added to the right. This matrix is left unchanged.
     *
     * @param numColumns Number of columns to add
     * @return The enlarged matrix
     */
    public Matrix addColumns(int numColumns) {
        if( numColumns < 0 )
            throw new IllegalArgumentException("Negative number of columns: " + numColumns);
        return appendRight(new Matrix(rows, numColumns));
    }

    /**
     * Returns a new matrix made by placing <code>m</code> to the right of this matrix.
     * Both matrices must have the same number of rows. Neither matrix is modified.
     * A common use is appending a column of ones to a feature matrix for a bias term.
     *
     * @param m The matrix to append
     * @return The combined matrix
     */
    public Matrix appendRight(Matrix m) {
        if( m.rows != rows )
            throw new IllegalArgumentException("Cannot append right: row counts differ: " + rows + " != " + m.rows);
        int cols = columns + m.columns;
        double[][] result = new double[rows][cols];
        for(int i=0; i<rows; i++) {
            System.arraycopy(matrix[i], 0, result[i], 0, columns);
            System.arraycopy(m.matrix[i], 0, result[i], columns, m.columns);
        }
        return adopt(result, cols);
    }

    /**
     * Returns a new matrix made by placing <code>m</code> below this matrix.
     * Both matrices must have the same number of columns. Neither matrix is modified.
     *
     * @param m The matrix to append
     * @return The combined matrix
     */
    public Matrix appendBottom(Matrix m) {
        if( m.columns != columns )
            throw new IllegalArgumentException("Cannot append bottom: column counts differ: " + columns + " != " + m.columns);
        double[][] result = new double[rows + m.rows][];
        for(int i=0; i<rows; i++)
            result[i] = matrix[i].clone();
        for(int i=0; i<m.rows; i++)
            result[rows + i] = m.matrix[i].clone();
        return adopt(result, columns);
    }

    // ------------------------------------------------------------------
    // In-place fillers
    // ------------------------------------------------------------------

    /**
     * Sets every element in this matrix (in place) to a random number between
     * -1 and 1.
     */
    public void randomize() {
        for(int i=0; i<rows; i++)
            for(int j=0; j<columns; j++)
                matrix[i][j] = Math.random() * 2 - 1;
    }

    /**
     * Sets every element in this matrix (in place) to 1.
     */
    public void ones() {
        fill(1);
    }

    /**
     * Sets every element in this matrix (in place) to 0.
     */
    public void zeros() {
        fill(0);
    }

    /**
     * Sets every element in this matrix (in place) equal to <code>x</code>.
     * @param x the value to fill the matrix
     */
    public void fill(double x) {
        for(int i=0; i<rows; i++)
            for(int j=0; j<columns; j++)
                matrix[i][j] = x;
    }

    // ------------------------------------------------------------------
    // Queries, comparison and access
    // ------------------------------------------------------------------

    /**
     * Returns true if the dimension of this matrix forms a square.
     *
     * @return Squared or not
     */
    public boolean isSquare() {
        return rows == columns;
    }

    private void requireSquare(String operation) {
        if( ! isSquare() )
            throw new IllegalStateException("Cannot compute " + operation + " of a non square matrix: " + rows + "x" + columns);
    }

    /**
     * Compares the supplied matrix <code>m</code> with the matrix
     * number by number. If all of the numbers are equal (and the
     * dimensions match) the method returns true and false otherwise.
     * Values are compared with <code>==</code>, so <code>0.0</code> equals
     * <code>-0.0</code> and a matrix containing NaN is never equal to anything.
     *
     * @param m The matrix to compare
     * @return Equal or not
     * @see #equals(Matrix, double)
     */
    public boolean equals(Matrix m) {
        return equals(m, 0);
    }

    /**
     * Compares the supplied matrix <code>m</code> with this matrix, treating two numbers as
     * equal when they differ by no more than <code>tolerance</code>. This is usually what you
     * want after floating point arithmetic, for example checking that
     * <code>a.dot(a.inverse())</code> is the identity.
     *
     * @param m The matrix to compare
     * @param tolerance The largest allowed absolute difference between corresponding elements
     * @return Equal within tolerance or not
     */
    public boolean equals(Matrix m, double tolerance) {
        if( m == null || rows != m.rows || columns != m.columns )
            return false;
        for(int i=0; i<rows; i++)
            for(int j=0; j<columns; j++)
                if( !(Math.abs(matrix[i][j] - m.matrix[i][j]) <= tolerance) )
                    return false;
        return true;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Matrix && equals((Matrix) o);
    }

    @Override
    public int hashCode() {
        int h = 31 * rows + columns;
        for(int i=0; i<rows; i++)
            for(int j=0; j<columns; j++)
                h = 31 * h + Double.hashCode(matrix[i][j] + 0.0); // + 0.0 folds -0.0 into 0.0
        return h;
    }

    /**
     * Returns a copy of the matrix.
     *
     * @return A copy of the matrix
     */
    public Matrix copy() {
        return adopt(deepCopy(matrix), columns);
    }

    /**
     * Return the value at the given row and column index.
     *
     * @param row The row
     * @param column The column
     * @return The value
     * @see #getMatrixArray()
     */
    public double getValueAt(int row, int column) {
        checkBounds(row, column);
        return matrix[row][column];
    }

    /**
     * Sets the value of the number at <code>row</code>, <code>column</code>
     * equal to <code>n</code>.
     *
     * @param row The row index
     * @param column The column index
     * @param n The new value
     */
    public void setValueAt(int row, int column, double n) {
        checkBounds(row, column);
        matrix[row][column] = n;
    }

    private void checkBounds(int row, int column) {
        if( row < 0 || row >= rows || column < 0 || column >= columns )
            throw new ArrayIndexOutOfBoundsException("Row and/or column out of bounds: "
                    + row + "," + column + " (matrix is " + rows + "x" + columns + ")");
    }

    /**
     * Converts the matrix into an array and returns the array. The conversion
     * is row-major: each row of this matrix is placed into the array in the
     * order in which the rows appear in the matrix.
     *
     * @return the array
     */
    public double[] toArray() {
        double[] result = new double[rows * columns];
        int k = 0;
        for(int i=0; i<rows; i++)
            for(int j=0; j<columns; j++)
                result[k++] = matrix[i][j];
        return result;
    }

    /**
     * Reads the values held in the supplied data
     * array into the matrix. Any data currently held by the matrix will be
     * lost. A new internal matrix array is created with the same dimensions
     * as <code>data</code> to contain the data. Note that the values are copied
     * in from <code>data</code> into
     * the matrix. No reference to <code>data</code> is held internally.
     *
     * @param data The data to read. Every row must have the same length.
     */
    public void readData(double data[][]) {
        load(data);
    }

    private void load(double[][] data) {
        if( data == null )
            throw new IllegalArgumentException("data must not be null");
        int cols = data.length == 0 ? 0 : data[0].length;
        double[][] copy = new double[data.length][];
        for(int i=0; i<data.length; i++) {
            if( data[i] == null || data[i].length != cols )
                throw new IllegalArgumentException("Row " + i + " has a different length from row 0 ("
                        + (data[i] == null ? "null" : data[i].length) + " != " + cols + ")");
            copy[i] = data[i].clone();
        }
        matrix = copy;
        rows = data.length;
        columns = cols;
    }

    /**
     * Returns the internal values of the matrix. Note that this is mutable. Changes
     * to the returned array will alter the internal array.
     *
     * @return the internal array of values
     */
    public double[][] getMatrixArray() {
        return matrix;
    }

    /**
     * Returns the dimension of the matrix as a Matrix object. The first element contains the
     * number of rows and the second contains the number of columns.
     * @return the dimension
     */
    public Matrix getDimension() {
        return adopt(new double[][] {{ rows, columns }}, 2);
    }

    /**
     * Return the number of rows.
     * @return rows
     */
    public int getRows() {
        return rows;
    }

    /**
     * Return the number of columns.
     * @return columns
     */
    public int getColumns() {
        return columns;
    }

    /**
     * Returns a printable, readable String representation of the
     * matrix, with the numbers in each column right-aligned.
     *
     * @return the matrix "string-ified"
     */
    @Override
    public String toString() {
        String[][] cells = new String[rows][columns];
        int[] width = new int[columns];
        for(int i=0; i<rows; i++)
            for(int j=0; j<columns; j++) {
                cells[i][j] = String.valueOf(matrix[i][j]);
                width[j] = Math.max(width[j], cells[i][j].length());
            }
        StringBuilder sb = new StringBuilder();
        for(int i=0; i<rows; i++) {
            for(int j=0; j<columns; j++) {
                if( j > 0 ) sb.append("  ");
                for(int p=cells[i][j].length(); p<width[j]; p++) sb.append(' ');
                sb.append(cells[i][j]);
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    /**
     * Prints the matrix to stdout.
     */
    public void print() {
        System.out.println(toString());
    }
}
