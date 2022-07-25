import unittest as ut
from lpsolver import *

#A = [[2,-1,2],[2,-3,1],[-1,1,-2]]
#b = [4,-5,-1]
#c = [1,-1,1]

# max lp tests
class TestMaxLps(ut.TestCase):
    def test_max_lp1(self):
        A = [[2, 3], [4,1], [1,1]]
        b = [120, 160, 43]
        c = [5, 4]
        prob = MaxLp(A,b,c)
        obj,_,_ = prob.solve()
        self.assertIsNotNone(obj)
        self.assertEqual(obj, 211.0)

    def test_max_lp2(self):
        A = [[2, -2], [0, 1]] 
        b = [3, 3]
        c = [4, -1]
        prob = MaxLp(A,b,c)
        obj,x,_ = prob.solve()
        self.assertIsNotNone(obj)
        self.assertEqual(obj, 15.0)
        self.assertEqual(x[0], 4.5)

    def test_dual_max_lp(self):
        A = [[2, 3], [-4,1], [1,1]]
        b = [120, -160, 43]
        c = [5, 4]
        prob = MaxLp(A,b,c)
        obj,x,_ = prob.solve()
        self.assertIsNotNone(obj)
        self.assertEqual(obj, 218.0)
        self.assertEqual(x[0], 43.6)

# min lp tests
class TestMinLps(ut.TestCase):
    def test_min_lp1(self):
        A = [[2, 3], [4,1], [1,1]]
        b = [120, 160, 43]
        c = [5, 4]
        prob = MinLp(A,b,c)
        obj,x,_ = prob.solve()
        self.assertIsNotNone(obj)
        self.assertEqual(obj, 0.0)
        self.assertEqual(x[0], 0)
        self.assertEqual(x[1], 0)

    def test_min_lp2(self):
        A = [[1, 0], [0,1], [-1,-1]]
        b = [20, 20, -10]
        c = [1, 1]
        prob = MinLp(A,b,c)
        obj,x,_ = prob.solve()
        self.assertIsNotNone(obj)
        self.assertEqual(obj, 20.0)
        self.assertEqual(x[0], 0)
        self.assertEqual(x[1], 20)

    def test_min_lp3(self):
        A = [[1, 0], [0,1], [-1,0], [0,-1]]
        b = [10, 10, -5, -5]
        c = [1, 1]
        prob = MinLp(A,b,c)
        obj,x,_ = prob.solve()
        self.assertIsNotNone(obj)
        self.assertEqual(obj, 10.0)
        self.assertEqual(x[0], 5)
        self.assertEqual(x[1], 5)


#max ip tests
class TestIpMax(ut.TestCase):
    @ut.skip("TODO: is broken")
    def test_max_ip(self):
        A = [[2, -2], [0, 1]] 
        b = [3, 3]
        c = [4, -1]
        prob = MaxIp(A,b,c)
        obj,x,_ = prob.solve()
        self.assertIsNotNone(obj)
        self.assertEquals(obj, 13.0)
        self.assertEquals(x[0], 4.0)
        self.assertEquals(x[0], 3.0)

# min ip tests
class TestIpMin(ut.TestCase):
    @ut.skip("TODO: is broken")
    def test_min_ip(self):
        A = [[2, -2], [0, 1]] 
        b = [3, 3]
        c = [4, -1]
        prob = MinIp(A,b,c)
        obj,x,_ = prob.solve()
        self.assertIsNotNone(obj)
        self.assertEquals(obj, 13.0)
        self.assertEquals(x[0], 4.0)
        self.assertEquals(x[0], 3.0)

# errors tests
class TestErrors(ut.TestCase):
    def test_invalid_base_error(self):
        A = np.array([[2,3,1,0,0], [4,1,0,1,0], [1,1,0,0,1]])
        b = np.array([120, -160, 43])
        c = np.array([5,4,0,0,0])
        with self.assertRaises(InfeasibleError):
            SIMPLEX(A,b,c)

    def test_shape_error(self):
        A = [[2, 3], [4,1]]
        b = [120, -160, 43]
        c = [5, 4]
        with self.assertRaises(ShapeError):
            prob = MaxLp(A,b,c)
    
    def test_shape_error2(self):
        A = [[2], [4], [5]]
        b = [120, -160, 43]
        c = [5, 4]
        with self.assertRaises(ShapeError):
            prob = MaxLp(A,b,c)

if __name__ == '__main__':
    ut.main()
