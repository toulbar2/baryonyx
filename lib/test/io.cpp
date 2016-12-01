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

#include <lpcore>
#include <lpcore-compare>
#include <lpcore-out>
#include <iostream>
#include <fstream>
#include <numeric>
#include <map>
#include <sstream>
#include "unit-test.hpp"

void test_examples_1()
{
    const char *example_1 = "maximize\n"
        "x1 + 2x2 + 3x3\n"
        "st\n"
        "time:  -x1 + x2 + x3 <= 20\n"
        "labor:  x1 - 3x2 + x3 <= 30\n"
        "bounds\n"
        "x1 <= 40\n"
        "end\n";

    std::istringstream iss(example_1);

    auto pb = lp::make_problem(iss);
    std::cout << __func__ << '\n' << lp::resume(pb) << '\n';

    Ensures(pb.type == lp::objective_function_type::maximize);
    Ensures(pb.vars.names.size() == 3);
    Ensures(pb.vars.values.size() == 3);

    Ensures(pb.vars.names[0] == "x1");
    Ensures(pb.vars.names[1] == "x2");
    Ensures(pb.vars.names[2] == "x3");

    Ensures(pb.vars.values[0].min == 0);
    Ensures(pb.vars.values[1].min == 0);
    Ensures(pb.vars.values[2].min == 0);

    Ensures(pb.vars.values[0].max == 40);
    Ensures(pb.vars.values[1].max == std::numeric_limits<int>::max());
    Ensures(pb.vars.values[2].max == std::numeric_limits<int>::max());
}

void test_examples_2()
{
    std::ifstream ifs;

    long loop[3] = { 5, 1, 4 };
    double results[3] = { 15, 21, 95 };
    std::vector<std::vector<int>> values(3);

    values[0] = { 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1 };
    values[1] = { 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0 };
    values[2] = { 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0 };

    for (int i = 1; i != 4; ++i) {
        std::string filepath { EXAMPLES_DIR "/assignment_problem_" };
        filepath += std::to_string(i);
        filepath += ".lp";

        auto pb = lp::make_problem(filepath);
        std::cout << __func__ << '\n' << lp::resume(pb) << '\n';

        Ensures(pb.vars.names.size() == 16);
        Ensures(pb.vars.values.size() == 16);

        std::stringstream ss;
        ss << pb;

        auto pb2 = lp::make_problem(ss);
        Ensures(pb == pb2);

        std::map<std::string, lp::parameter> params;
        params["kappa"] = 0.5;
        params["theta"] = 0.5;
        params["delta"] = 0.5;
        params["limit"] = 10l;

        auto result = lp::solve(pb, params);

        Ensures(result.optimal == true);
        Ensures(result.loop == loop[i - 1]);
        Ensures(result.value == results[i - 1]);
        Ensures(result.variable_value == values[i - 1]);

        std::cout << result << '\n';
    }
}

void test_examples_3()
{
    auto pb = lp::make_problem(EXAMPLES_DIR
                               "/geom-30a-3-ext_1000_support.lp");
    std::cout << __func__ << '\n' << lp::resume(pb) << '\n';

    Ensures(pb.type == lp::objective_function_type::minimize);
    Ensures(pb.vars.names.size() == 819);
    Ensures(pb.vars.values.size() == 819);

    lp::index nb {0};
    for (auto& elem : pb.vars.values)
        if (elem.type == lp::variable_type::binary)
            ++nb;

    Ensures(nb == 90);
}

void test_examples_4()
{
    auto pb = lp::make_problem(EXAMPLES_DIR "/general.lp");
    std::cout << __func__ << '\n' << lp::resume(pb) << '\n';

    Ensures(pb.type == lp::objective_function_type::minimize);
    Ensures(pb.vars.names.size() == 3);
    Ensures(pb.vars.values.size() == 3);

    lp::index nb {0};
    for (auto& elem : pb.vars.values)
        if (elem.type == lp::variable_type::general)
            ++nb;

    Ensures(nb == 3);

    std::map<std::string, lp::parameter> params;
    params["kappa"] = 0.5;
    params["theta"] = 0.5;
    params["delta"] = 0.5;
    params["limit"] = 10l;

    auto result = lp::solve(pb, params);

    std::cout << result << '\n';
}

void test_examples_sudoku()
{
    auto pb = lp::make_problem(EXAMPLES_DIR "/sudoku.lp");

    Ensures(pb.vars.names.size() == 81);
    Ensures(pb.vars.values.size() == 81);

    for (auto& vv : pb.vars.values) {
        Ensures(vv.min == 1);
        Ensures(vv.max == 9);
        Ensures(vv.min_equal == true);
        Ensures(vv.max_equal == true);
        Ensures(vv.type == lp::variable_type::general);
    }

    std::cout << __func__ << '\n' << lp::resume(pb) << '\n';
}

void test_examples_8_queens_puzzle()
{
    auto pb = lp::make_problem(EXAMPLES_DIR "/8_queens_puzzle.lp");

    Ensures(pb.vars.names.size() == 64);
    Ensures(pb.vars.values.size() == 64);

    for (auto& vv : pb.vars.values) {
        Ensures(vv.min == 0);
        Ensures(vv.max == 1);
        Ensures(vv.min_equal == true);
        Ensures(vv.max_equal == true);
        Ensures(vv.type == lp::variable_type::binary);
    }

    std::cout << __func__ << '\n' << lp::resume(pb) << '\n';
}

void test_examples_vm()
{
    auto pb = lp::make_problem(EXAMPLES_DIR "/vm.lp");

    std::cout << __func__ << '\n' << lp::resume(pb) << '\n';
}

int main(int /* argc */, char */* argv */[])
{
    test_examples_1();
    test_examples_2();
    test_examples_3();
    test_examples_4();
    test_examples_sudoku();
    test_examples_8_queens_puzzle();
    test_examples_vm();

    return unit_test::report_errors();
}
