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
     * Computes the determinant of the matrix using Gaussian elimination with partial
     * pivoting. The matrix must be square and is left unchanged.
     *
     * @return the determinant
     */
    public double determinant() {
        requireSquare("determinant");
        if( rows == 0 )
            return 1; // by convention, the determinant of a 0x0 matrix is 1
        int n = rows;
        double[][] a = deepCopy(matrix);
        double det = 1;
        for(int col=0; col<n; col++) {
            int pivot = col;
            for(int r=col+1; r<n; r++)
                if( Math.abs(a[r][col]) > Math.abs(a[pivot][col]) )
                    pivot = r;
            if( a[pivot][col] == 0 )
                return 0;
            if( pivot != col ) {
                double[] tmp = a[pivot]; a[pivot] = a[col]; a[col] = tmp;
                det = -det;
            }
            det *= a[col][col];
            for(int r=col+1; r<n; r++) {
                double f = a[r][col] / a[col][col];
                for(int c=col; c<n; c++)
                    a[r][c] -= f * a[col][c];
            }
        }
        return det;
    }

    // ------------------------------------------------------------------
    // Transpose, inverse and solving
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
     * Computes the inverse of the matrix. This matrix is left unchanged.
     * <p>
     * If you only need the inverse in order to compute <code>inverse().dot(b)</code>,
     * call <code>solve(b)</code> instead: it is faster and more accurate.
     *
     * @return The inverse
     * @exception NonInvertibleMatrixException If the matrix is not square, is empty,
     * or is singular (or so close to singular that its inverse would be meaningless)
     * @see #solve(Matrix)
     */
    public Matrix inverse() throws NonInvertibleMatrixException {
        if( ! isSquare() )
            throw new NonInvertibleMatrixException("Cannot invert a non square matrix: " + rows + "x" + columns);
        return solve(identity(rows));
    }

    /**
     * Solves the linear system <code>A*X = B</code> for <code>X</code>, where A is this
     * (square) matrix and B is the supplied matrix <code>rhs</code>. B may have any number of
     * columns; each column is solved as a separate right hand side. Neither matrix is
     * modified.
     *
     * @param rhs The right hand side B. It must have the same number of rows as this matrix.
     * @return X, a matrix with the same dimensions as <code>rhs</code>
     * @exception NonInvertibleMatrixException If this matrix is not square, is empty or is singular
     */
    public Matrix solve(Matrix rhs) throws NonInvertibleMatrixException {
        if( ! isSquare() )
            throw new NonInvertibleMatrixException("Cannot solve with a non square matrix: " + rows + "x" + columns);
        if( rows == 0 )
            throw new NonInvertibleMatrixException("Cannot solve with an empty matrix.");
        if( rhs.rows != rows )
            throw new IllegalArgumentException("Right hand side must have " + rows +
                    " rows but has " + rhs.rows);

        int n = rows;
        int m = rhs.columns;
        // Work on copies: gaussian() eliminates in place and must never touch our data
        double[][] a = deepCopy(matrix);
        double[][] b = deepCopy(rhs.matrix);
        int[] index = new int[n];

        // Transform a copy of the matrix into an upper triangle
        gaussian(a, index);

        // Apply the stored elimination ratios to the right hand side
        for (int i=0; i<n-1; ++i)
            for (int j=i+1; j<n; ++j)
                for (int k=0; k<m; ++k)
                    b[index[j]][k] -= a[index[j]][i]*b[index[i]][k];

        // Perform backward substitutions
        double[][] x = new double[n][m];
        for (int col=0; col<m; ++col) {
            x[n-1][col] = b[index[n-1]][col]/a[index[n-1]][n-1];
            for (int j=n-2; j>=0; --j) {
                double v = b[index[j]][col];
                for (int k=j+1; k<n; ++k)
                    v -= a[index[j]][k]*x[k][col];
                x[j][col] = v / a[index[j]][j];
            }
        }
        return adopt(x, m);
    }

    /**
     * Gaussian elimination with scaled partial pivoting, performed IN PLACE on
     * <code>a</code>. Callers must pass a copy. On return the upper triangle of
     * <code>a</code> holds U, the elimination ratios are stored below the diagonal,
     * and <code>index</code> holds the pivoting order.
     */
    private static void gaussian(double[][] a, int[] index) throws NonInvertibleMatrixException {
        int n = index.length;
        double[] c = new double[n];

        for (int i=0; i<n; ++i)
            index[i] = i;

        // Find the rescaling factors, one from each row
        for (int i=0; i<n; ++i) {
            double c1 = 0;
            for (int j=0; j<n; ++j) {
                double c0 = Math.abs(a[i][j]);
                if (c0 > c1) c1 = c0;
            }
            if (c1 == 0)
                throw new NonInvertibleMatrixException("Matrix is singular: row " + i + " is all zeros.");
            c[i] = c1;
        }

        for (int j=0; j<n-1; ++j) {
            // Search for the pivoting element in this column
            double pi1 = 0;
            int k = j;
            for (int i=j; i<n; ++i) {
                double pi0 = Math.abs(a[index[i]][j]) / c[index[i]];
                if (pi0 > pi1) {
                    pi1 = pi0;
                    k = i;
                }
            }
            if (pi1 < SINGULAR_TOLERANCE)
                throw new NonInvertibleMatrixException("Matrix is singular (no usable pivot in column " + j + ").");

            // Interchange rows according to the pivoting order
            int itmp = index[j];
            index[j] = index[k];
            index[k] = itmp;

            for (int i=j+1; i<n; ++i) {
                double pj = a[index[i]][j]/a[index[j]][j];
                // Record pivoting ratios below the diagonal
                a[index[i]][j] = pj;
                // Modify other elements accordingly
                for (int l=j+1; l<n; ++l)
                    a[index[i]][l] -= pj*a[index[j]][l];
            }
        }

        int last = index[n-1];
        if (Math.abs(a[last][n-1]) / c[last] < SINGULAR_TOLERANCE)
            throw new NonInvertibleMatrixException("Matrix is singular (no usable pivot in column " + (n-1) + ").");
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
