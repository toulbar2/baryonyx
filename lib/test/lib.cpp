/* Copyright (C) 2016 INRA
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "fixed_2darray.hpp"
#include "fixed_array.hpp"
#include "knapsack-solver.hpp"
#include "matrix.hpp"
#include "scoped_array.hpp"
#include "unit-test.hpp"

#include <baryonyx/core>

#include <functional>
#include <iostream>
#include <numeric>

void
check_clamp()
{
    Ensures(baryonyx::clamp(0.0, 0.0, 1.0) == 0.0);
    Ensures(baryonyx::clamp(1.0, 0.0, 1.0) == 1.0);
    Ensures(baryonyx::clamp(-0.5, 0.0, 1.0) == 0.0);
    Ensures(baryonyx::clamp(1.5, 0.0, 1.0) == 1.0);
    Ensures(baryonyx::clamp(168, -128, +127) == 127);
    Ensures(baryonyx::clamp(168, 0, +255) == 168);
    Ensures(baryonyx::clamp(128, -128, +127) == 127);
    Ensures(baryonyx::clamp(128, 0, +255) == 128);
}

void
check_numeric_cast()
{
    int small_positive = 1;
    int small_negative = -1;
    int large_positive = std::numeric_limits<int>::max();
    int large_negative = std::numeric_limits<int>::min();

    Ensures(baryonyx::is_numeric_castable<signed char>(small_positive));
    Ensures(baryonyx::is_numeric_castable<signed char>(small_negative));
    Ensures(not baryonyx::is_numeric_castable<signed char>(large_positive));
    Ensures(not baryonyx::is_numeric_castable<signed char>(large_negative));

    Ensures(baryonyx::is_numeric_castable<unsigned char>(small_positive));
    Ensures(not baryonyx::is_numeric_castable<unsigned char>(small_negative));
    Ensures(not baryonyx::is_numeric_castable<unsigned char>(large_positive));
    Ensures(not baryonyx::is_numeric_castable<unsigned char>(large_negative));

    Ensures(baryonyx::is_numeric_castable<signed int>(small_positive));
    Ensures(baryonyx::is_numeric_castable<signed int>(small_negative));
    Ensures(baryonyx::is_numeric_castable<signed int>(large_positive));
    Ensures(baryonyx::is_numeric_castable<signed int>(large_negative));

    Ensures(baryonyx::is_numeric_castable<unsigned int>(small_positive));
    Ensures(not baryonyx::is_numeric_castable<unsigned int>(small_negative));
    Ensures(baryonyx::is_numeric_castable<unsigned int>(large_positive));
    Ensures(not baryonyx::is_numeric_castable<unsigned int>(large_negative));

    Ensures(baryonyx::is_numeric_castable<long long>(small_positive));
    Ensures(baryonyx::is_numeric_castable<long long>(large_negative));
    Ensures(baryonyx::is_numeric_castable<long long>(small_positive));
    Ensures(baryonyx::is_numeric_castable<long long>(large_negative));

    Ensures(baryonyx::is_numeric_castable<unsigned long long>(small_positive));
    Ensures(
      not baryonyx::is_numeric_castable<unsigned long long>(small_negative));
    Ensures(baryonyx::is_numeric_castable<unsigned long long>(large_positive));
    Ensures(
      not baryonyx::is_numeric_castable<unsigned long long>(large_negative));

    Ensures(not baryonyx::is_numeric_castable<size_t>(small_negative));
    Ensures(not baryonyx::is_numeric_castable<size_t>(large_negative));

    std::vector<int> v;

    EnsuresNotThrow(baryonyx::numeric_cast<short int>(v.size()),
                    std::exception);

    EnsuresNotThrow(baryonyx::numeric_cast<short int>(v.capacity()),
                    std::exception);

    EnsuresThrow(baryonyx::numeric_cast<short int>(v.max_size()),
                 std::exception);

    auto checked_size = baryonyx::numeric_cast<unsigned int>(v.size());
    Ensures(0 == checked_size);
}

void
check_parameter()
{
    baryonyx::parameter real{ 3.0 };
    Ensures(real.type == baryonyx::parameter::tag::real);

    baryonyx::parameter integer{ 1000 };
    Ensures(integer.type == baryonyx::parameter::tag::integer);

    baryonyx::parameter str{ "hello world" };
    Ensures(str.type == baryonyx::parameter::tag::string);

    str = real;
    Ensures(str.type == baryonyx::parameter::tag::real);
    Ensures(str.d == 3.0);

    str = integer;
    Ensures(str.type == baryonyx::parameter::tag::integer);
    Ensures(str.l == 1000);

    std::vector<baryonyx::parameter> x(100);
    for (auto& elem : x) {
        Ensures(elem.type == baryonyx::parameter::tag::integer);
        Ensures(elem.l == 0);
    }

    auto y = baryonyx::parameter(4.0);
    Ensures(y.type == baryonyx::parameter::tag::real);
    Ensures(y.d == 4.0);

    x[0] = baryonyx::parameter(5.0);
    Ensures(x[0].type == baryonyx::parameter::tag::real);
    Ensures(x[0].d == 5.0);

    x[0].swap(x[1]);
    Ensures(x[0].type == baryonyx::parameter::tag::integer);
    Ensures(x[0].l == 0l);
    Ensures(x[1].type == baryonyx::parameter::tag::real);
    Ensures(x[1].d == 5.0);

    x[2] = std::move(x[1]);
    Ensures(x[0].type == baryonyx::parameter::tag::integer);
    Ensures(x[0].l == 0l);
    Ensures(x[1].type == baryonyx::parameter::tag::integer);
    Ensures(x[1].l == 0l);
    Ensures(x[2].type == baryonyx::parameter::tag::real);
    EnsuresEqual(x[2].d, 5.0);
    EnsuresNotEqual(x[2].d, 6.0);

    x[3] = baryonyx::parameter(std::string("hello world!"));
    Ensures(x[3].type == baryonyx::parameter::tag::string);
    Ensures(x[3].s == "hello world!");
}

std::ptrdiff_t
size(std::tuple<baryonyx::SparseArray<int, double>::const_iterator,
                baryonyx::SparseArray<int, double>::const_iterator> elem)
{
    return std::distance(std::get<0>(elem), std::get<1>(elem));
}

void
check_matrix()
{
    std::vector<int> row{ 1, 1, 1, 1 };
    std::vector<int> col{ 1, 3 };

    baryonyx::SparseArray<int, double> m(4, 2);
    m.reserve(4, row.begin(), row.end(), col.begin(), col.end());

    EnsuresThrow(m.P(0, 0), std::out_of_range);
    EnsuresThrow(m.P(0, 1), std::out_of_range);
    EnsuresThrow(m.P(1, 0), std::out_of_range);
    EnsuresThrow(m.P(1, 1), std::out_of_range);
    EnsuresThrow(m.P(2, 0), std::out_of_range);
    EnsuresThrow(m.P(2, 1), std::out_of_range);
    EnsuresThrow(m.P(3, 0), std::out_of_range);
    EnsuresThrow(m.P(3, 1), std::out_of_range);
    Ensures(m.size() == 0);

    m.set(1, 0, 1, 1.0);
    m.set(0, 1, 2, 2.0);
    m.set(3, 1, 3, 3.0);
    m.set(2, 1, 4, 4.0);
    m.sort();

    std::cout << m << '\n';

    Ensures(m.size() == 4);
    EnsuresThrow(m.P(0, 0), std::out_of_range);
    Ensures(m.A(0, 1) == 2);
    Ensures(m.P(0, 1) == 2.0);

    Ensures(m.A(1, 0) == 1);
    Ensures(m.P(1, 0) == 1.0);
    EnsuresThrow(m.A(1, 1), std::out_of_range);
    EnsuresThrow(m.A(2, 0), std::out_of_range);
    Ensures(m.A(2, 1) == 4);
    Ensures(m.P(2, 1) == 4.0);
    EnsuresThrow(m.A(3, 0), std::out_of_range);
    Ensures(m.A(3, 1) == 3);
    Ensures(m.P(3, 1) == 3.0);
    Ensures(m.size() == 4);

    Ensures(size(m.row(0)) == 1);
    Ensures(size(m.row(1)) == 1);
    Ensures(size(m.row(2)) == 1);
    Ensures(size(m.row(3)) == 1);
    Ensures(size(m.column(0)) == 1);

    Ensures(size(m.column(1)) == 3);

    Ensures(m.A().size() == 4);
    Ensures(m.P()[0] == 1.0);
    Ensures(m.P()[1] == 2.0);
    Ensures(m.P()[2] == 3.0);
    Ensures(m.P()[3] == 4.0);
}

void
check_scoped_array()
{
    baryonyx::scoped_array<int> a(10);
    if (not a) {
        Ensures(a);
        return;
    }

    int* array = a.data();

    Ensures(a.data() == array);

    array[0] = 1;
    array[1] = 2;
    array[2] = 3;
    array[3] = 4;
    array[4] = 5;
    array[5] = 6;
    array[6] = 7;
    array[7] = 8;
    array[8] = 9;
    array[9] = 0;

    Ensures(a[0] == 1);
    Ensures(a[1] == 2);
    Ensures(a[2] == 3);
    Ensures(a[3] == 4);
    Ensures(a[4] == 5);
    Ensures(a[5] == 6);
    Ensures(a[6] == 7);
    Ensures(a[7] == 8);
    Ensures(a[8] == 9);
    Ensures(a[9] == 0);

    baryonyx::scoped_array<int> b(std::move(a));

    Ensures(b.data() == array);
    Ensures(a.data() == nullptr);

    std::swap(a, b);

    Ensures(a.data() == array);
    Ensures(b.data() == nullptr);

    baryonyx::scoped_array<double> x(1000, 123.0);
    if (x) {
        for (int i{ 0 }; i < 1000; ++i)
            Ensures(x[i] == 123.0);
    }
}

void
check_fixed_array()
{
    baryonyx::fixed_array<int> a(10);

    Ensures(a.size() == 10);

    std::iota(a.begin(), a.end(), 1);

    Ensures(a[0] == 1);
    Ensures(a[1] == 2);
    Ensures(a[2] == 3);
    Ensures(a[3] == 4);
    Ensures(a[4] == 5);
    Ensures(a[5] == 6);
    Ensures(a[6] == 7);
    Ensures(a[7] == 8);
    Ensures(a[8] == 9);
    Ensures(a[9] == 10);

    {
        baryonyx::fixed_array<int> copy(a);

        std::iota(copy.rbegin(), copy.rend(), 1);
        Ensures(copy[9] == 1);
        Ensures(copy[8] == 2);
        Ensures(copy[7] == 3);
        Ensures(copy[6] == 4);
        Ensures(copy[5] == 5);
        Ensures(copy[4] == 6);
        Ensures(copy[3] == 7);
        Ensures(copy[2] == 8);
        Ensures(copy[1] == 9);
        Ensures(copy[0] == 10);
    }

    baryonyx::fixed_array<int> b(a);

    Ensures(a.data() != b.data());

    baryonyx::fixed_array<int> c(std::move(a));

    Ensures(a.data() == nullptr);
    Ensures(b.data() != c.data());

    baryonyx::fixed_array<double> d(15, 3.0);

    Ensures(d[0] == 3.0);
    Ensures(d[7] == 3.0);
    Ensures(d[14] == 3.0);

    baryonyx::fixed_array<double> e;

    std::swap(d, e);

    Ensures(not d);
    Ensures(e[0] == 3.0);
    Ensures(e[7] == 3.0);
    Ensures(e[14] == 3.0);

    baryonyx::fixed_array<double> x(1000, 123.0);
    if (x) {
        for (int i{ 0 }; i < 1000; ++i)
            Ensures(x[i] == 123.0);

        auto it = std::find_if_not(
          x.begin(),
          x.end(),
          std::bind(std::equal_to<double>(), 123.0, std::placeholders::_1));

        Ensures(it == x.end());
    }
}

void
check_fixed_2darray()
{
    baryonyx::fixed_2darray<int> a(2, 10);

    Ensures(a.size() == 20);
    Ensures(a.rows() == 2);
    Ensures(a.columns() == 10);

    std::iota(a.begin(), a.end(), 0);

    Ensures(a.data()[0] == 0);
    Ensures(a.data()[19] == 19);

    std::iota(a.rbegin(), a.rend(), 0);

    Ensures(a.data()[0] == 19);
    Ensures(a.data()[19] == 0);

    Ensures(a(0, 0) == 19);
    Ensures(a(1, 9) == 0);

    a(0, 1) = 100;
    a(1, 0) = 200;

    Ensures(a.data()[1] == 100);
    Ensures(a.data()[10] == 200);
}

void
check_knapsack_solver()
{
    // An example:
    // maximize: 16x1 + 19x2 + 23x3 + 26x4
    // st: 2x1 + 3x2 +4x3 + 5x4 <= 7

    std::vector<int> w{ 2, 3, 4, 5 };
    std::vector<int> v{ 16, 19, 23, 28 };
    int capacity{ 7 };

    auto result = baryonyx::knapsack_solver_dp<int>(v, w, capacity);

    Ensures(result.size() == 4);
    Ensures(result[0] == true);
    Ensures(result[1] == false);
    Ensures(result[2] == false);
    Ensures(result[3] == true);
}

int
main(int /* argc */, char* /* argv */ [])
{
    check_clamp();
    check_numeric_cast();
    check_parameter();
    check_matrix();
    check_scoped_array();
    check_fixed_array();
    check_fixed_2darray();
    check_knapsack_solver();

    return unit_test::report_errors();
}
