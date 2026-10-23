import java.util.Random;

/**
 * Self-contained tests for Matrix (no JUnit needed).
 * Run with:  javac *.java && java MatrixTest
 */
public class MatrixTest {
    private static int passed = 0, failed = 0;

    private static void check(String name, boolean ok) {
        if (ok) passed++;
        else { failed++; System.out.println("FAIL: " + name); }
    }

    private static void expectThrows(String name, Class<? extends Throwable> type, ThrowingRunnable r) {
        try { r.run(); check(name + " (no exception thrown)", false); }
        catch (Throwable t) { check(name + " (threw " + t + ")", type.isInstance(t)); }
    }

    interface ThrowingRunnable { void run() throws Exception; }

    private static Matrix m(double[][] d) { return new Matrix(d); }

    public static void main(String[] args) throws Exception {
        final double TOL = 1e-9;

        // --- The headline bug: inverse() must not mutate the original ---
        Matrix a = m(new double[][] {{1,2},{3,4}});
        Matrix before = a.copy();
        Matrix inv1 = a.inverse();
        check("inverse leaves original unchanged", a.equals(before));
        Matrix inv2 = a.inverse();
        check("repeated inverse gives the same answer", inv1.equals(inv2, TOL));
        check("inverse value", inv1.equals(m(new double[][] {{-2,1},{1.5,-0.5}}), TOL));
        check("A * inv(A) = I", a.dot(inv1).equals(Matrix.identity(2), TOL));

        // --- Singular matrices are reported, not turned into Infinity/NaN ---
        expectThrows("singular 2x2", NonInvertibleMatrixException.class,
                () -> m(new double[][] {{1,2},{2,4}}).inverse());
        expectThrows("zero row", NonInvertibleMatrixException.class,
                () -> m(new double[][] {{1,2},{0,0}}).inverse());
        expectThrows("singular 3x3 (rows dependent)", NonInvertibleMatrixException.class,
                () -> m(new double[][] {{1,2,3},{4,5,6},{7,8,9}}).inverse());
        expectThrows("1x1 zero", NonInvertibleMatrixException.class,
                () -> m(new double[][] {{0}}).inverse());
        expectThrows("non square", NonInvertibleMatrixException.class,
                () -> m(new double[][] {{1,2,3},{4,5,6}}).inverse());
        expectThrows("empty", NonInvertibleMatrixException.class, () -> new Matrix().inverse());
        check("1x1 inverse", m(new double[][] {{4}}).inverse().equals(m(new double[][] {{0.25}}), TOL));
        // Needs a row swap: first pivot is zero
        Matrix swap = m(new double[][] {{0,1},{1,0}});
        check("inverse with zero leading pivot", swap.inverse().equals(swap, TOL));

        // Random well-conditioned matrices, several sizes
        Random rnd = new Random(42);
        for (int n = 2; n <= 8; n++) {
            Matrix r = new Matrix(n, n);
            r.randomize();
            Matrix d = r.add(Matrix.identity(n).multiply(n)); // diagonally dominant => invertible
            Matrix dCopy = d.copy();
            check("random " + n + "x" + n + " A*inv(A)=I", d.dot(d.inverse()).equals(Matrix.identity(n), 1e-9));
            check("random " + n + "x" + n + " untouched", d.equals(dCopy));
        }

        // --- solve ---
        Matrix A = m(new double[][] {{2,1,-1},{-3,-1,2},{-2,1,2}});
        Matrix b = Matrix.fromArray(new double[] {8,-11,-3});
        Matrix x = A.solve(b);
        check("solve classic system", x.equals(Matrix.fromArray(new double[] {2,3,-1}), TOL));
        check("solve leaves A unchanged", A.equals(m(new double[][] {{2,1,-1},{-3,-1,2},{-2,1,2}})));
        check("solve leaves b unchanged", b.equals(Matrix.fromArray(new double[] {8,-11,-3})));
        check("solve with identity = inverse", A.solve(Matrix.identity(3)).equals(A.inverse(), TOL));
        expectThrows("solve rhs wrong rows", IllegalArgumentException.class, () -> A.solve(new Matrix(2, 1)));

        // --- determinant and trace ---
        check("det 2x2", Math.abs(a.determinant() - (-2)) < TOL);
        check("det 3x3", Math.abs(A.determinant() - (-1)) < TOL);
        check("det singular", Math.abs(m(new double[][] {{1,2,3},{4,5,6},{7,8,9}}).determinant()) < TOL);
        check("det needs swap", Math.abs(swap.determinant() - (-1)) < TOL);
        check("det leaves original unchanged", a.equals(before));
        check("trace", A.trace() == 2 - 1 + 2);
        expectThrows("trace non square", IllegalStateException.class, () -> new Matrix(2,3).trace());

        // --- dot vs multiply ---
        Matrix p = m(new double[][] {{1,3,2},{4,0,1}});
        Matrix q = m(new double[][] {{1,3},{0,1},{5,2}});
        check("dot", p.dot(q).equals(m(new double[][] {{11,10},{9,14}})));
        check("hadamard", m(new double[][] {{1,0},{0,1}}).multiply(m(new double[][] {{5,6},{7,8}}))
                .equals(m(new double[][] {{5,0},{0,8}})));
        check("dot into zero-row matrix", new Matrix(2,0).dot(new Matrix(0,3)).equals(new Matrix(2,3)));
        expectThrows("dot bad dims", IllegalArgumentException.class, () -> p.dot(p));

        // --- arithmetic returns new matrices ---
        Matrix orig = m(new double[][] {{1,2},{3,4}});
        Matrix origCopy = orig.copy();
        orig.add(1); orig.subtract(1); orig.multiply(2); orig.divide(2); orig.square();
        orig.add(orig); orig.subtract(orig); orig.multiply(orig); orig.map(v -> -v);
        orig.transpose(); orig.appendRight(orig); orig.appendBottom(orig); orig.addRows(1); orig.addColumns(1);
        check("no operation mutates the receiver", orig.equals(origCopy));
        check("square", orig.square().equals(m(new double[][] {{1,4},{9,16}})));
        check("mean", orig.mean() == 2.5);
        check("static map", Matrix.map(orig, v -> v * 10).equals(m(new double[][] {{10,20},{30,40}})));

        // --- structural features ---
        check("identity()", m(new double[][] {{5,5},{5,5}}).identity().equals(Matrix.identity(2)));
        check("getRow", orig.getRow(1).equals(m(new double[][] {{3,4}})));
        check("getColumn", orig.getColumn(1).equals(Matrix.fromArray(new double[] {2,4})));
        check("getColumn last index works", orig.getColumn(1).getRows() == 2);
        expectThrows("getColumn out of range", IllegalArgumentException.class, () -> orig.getColumn(2));
        expectThrows("getRow out of range", IllegalArgumentException.class, () -> orig.getRow(2));
        expectThrows("getValueAt column == columns", ArrayIndexOutOfBoundsException.class, () -> orig.getValueAt(0, 2));
        check("appendRight", orig.appendRight(Matrix.fromArray(new double[] {9,9}))
                .equals(m(new double[][] {{1,2,9},{3,4,9}})));
        check("appendBottom", orig.appendBottom(m(new double[][] {{7,8}}))
                .equals(m(new double[][] {{1,2},{3,4},{7,8}})));
        check("addRows", orig.addRows(2).equals(m(new double[][] {{1,2},{3,4},{0,0},{0,0}})));
        check("addColumns", orig.addColumns(1).equals(m(new double[][] {{1,2,0},{3,4,0}})));
        expectThrows("appendRight mismatch", IllegalArgumentException.class, () -> orig.appendRight(new Matrix(3,1)));
        check("getAsVector column", orig.getAsVector(Matrix.COLUMN_VECTOR).equals(Matrix.fromArray(new double[] {1,2,3,4})));
        check("getAsVector row", orig.getAsVector(Matrix.ROW_VECTOR).equals(m(new double[][] {{1,2,3,4}})));

        // --- appendBottom result rows are independent of the inputs ---
        Matrix stacked = orig.appendBottom(orig);
        stacked.setValueAt(0, 0, 99);
        check("appendBottom copies rows", orig.getValueAt(0, 0) == 1);

        // --- readData / construction robustness ---
        expectThrows("jagged data", IllegalArgumentException.class, () -> m(new double[][] {{1,2},{3}}));
        check("readData empty array", new Matrix(new double[0][0]).getRows() == 0);
        check("copy of empty matrix", new Matrix().copy().getRows() == 0);
        double[][] src = {{1,2}};
        Matrix fromSrc = m(src);
        src[0][0] = 100;
        check("constructor copies data", fromSrc.getValueAt(0, 0) == 1);

        // --- equals / hashCode contract ---
        Matrix z1 = m(new double[][] {{0.0}}), z2 = m(new double[][] {{-0.0}});
        check("0.0 equals -0.0", z1.equals(z2));
        check("equal matrices share hashCode (incl. -0.0)", z1.hashCode() == z2.hashCode());
        check("equals(Object)", ((Object) orig).equals(origCopy));
        check("equals null", !orig.equals((Matrix) null));
        check("equals with tolerance", orig.equals(orig.add(1e-12), 1e-9) && !orig.equals(orig.add(1e-12)));

        // --- toString alignment ---
        String s = m(new double[][] {{1,-20.5},{300,4}}).toString();
        check("toString right-aligns", s.equals("  1.0  -20.5\n300.0    4.0\n"));

        // --- MathUtils and Vector fixes ---
        check("inRange is static and inclusive", MathUtils.inRange(10, 0, 10) && !MathUtils.inRange(10.1, 0, 10));
        boolean floatsOk = true;
        for (int i = 0; i < 1000; i++) {
            float f = MathUtils.nextFloat(2f, 3f);   // compiles only because it returns float
            if (f < 2f || f >= 3f) floatsOk = false;
        }
        check("nextFloat returns float in range", floatsOk);
        Vector v = new Vector(1, 2, 3);
        v.set(4, 5);
        check("Vector.set(x,y) zeroes z", v.x == 4 && v.y == 5 && v.z == 0);

        System.out.println(passed + " passed, " + failed + " failed");
        if (failed > 0) System.exit(1);
    }
}
