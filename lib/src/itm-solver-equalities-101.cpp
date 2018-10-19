/* Copyright (C) 2016-2018 INRA
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

#include "itm-common.hpp"
#include "itm-optimizer-common.hpp"
#include "itm-solver-common.hpp"
#include "sparse-vector.hpp"

namespace baryonyx {
namespace itm {

template<typename Float, typename Mode, typename Random>
struct solver_equalities_101coeff
{
    using mode_type = Mode;
    using float_type = Float;

    Random& rng;

    struct rc_data
    {
        Float value;
        int id;
        bool is_negative;
    };

    struct rc_size
    {
        int r_size;
        int c_size;
    };

    sparse_matrix<int> ap;
    std::unique_ptr<Float[]> P;
    std::unique_ptr<int[]> A;
    std::unique_ptr<rc_data[]> R;
    std::unique_ptr<int[]> b;
    std::unique_ptr<Float[]> pi;

    const std::unique_ptr<Float[]>& c;

    int m;
    int n;

    solver_equalities_101coeff(Random& rng_,
                               int m_,
                               int n_,
                               const std::unique_ptr<Float[]>& c_,
                               const std::vector<merged_constraint>& csts)
      : rng(rng_)
      , ap(csts, m_, n_)
      , P(std::make_unique<Float[]>(ap.size()))
      , A(std::make_unique<int[]>(ap.size()))
      , R(std::make_unique<rc_data[]>(compute_reduced_costs_vector_size(csts)))
      , b(std::make_unique<int[]>(m_))
      , pi(std::make_unique<Float[]>(m_))
      , c(c_)
      , m(m_)
      , n(n_)
    {
        int id = 0;
        for (int i = 0; i != m; ++i) {
            for (const auto& cst : csts[i].elements) {
                bx_ensures(std::abs(cst.factor) == 1);
                A[id++] = cst.factor;
            }

            bx_ensures(csts[i].min == csts[i].max);

            b[i] = csts[i].min;
        }
    }

    int factor(int value) const noexcept
    {
        return A[value];
    }

    int bound_min(int constraint) const noexcept
    {
        return b[constraint];
    }

    int bound_max(int constraint) const noexcept
    {
        return b[constraint];
    }

    int bound_init(int constraint) const
    {
        return b[constraint];
    }

    Float compute_sum_A_pi(int variable) const
    {
        Float ret{ 0 };

        sparse_matrix<int>::const_col_iterator ht, hend;
        std::tie(ht, hend) = ap.column(variable);

        for (; ht != hend; ++ht)
            ret += pi[ht->row];

        return ret;
    }

    template<typename Xtype>
    bool is_valid_solution(const Xtype& x) const
    {
        for (int k = 0; k != m; ++k) {
            typename sparse_matrix<int>::const_row_iterator it, et;

            std::tie(it, et) = ap.row(k);
            int v = 0;

            for (; it != et; ++it)
                v += A[it->value] * x[it->column];

            if (b[k] != v)
                return false;
        }

        return true;
    }

    template<typename Xtype>
    bool is_valid_constraint(int k, const Xtype& x) const
    {
        typename sparse_matrix<int>::const_row_iterator it, et;

        std::tie(it, et) = ap.row(k);
        int v = 0;

        for (; it != et; ++it)
            v += A[it->value] * x[it->column];

        return b[k] == v;
    }

    template<typename Xtype>
    int compute_violated_constraints(const Xtype& x,
                                     std::vector<int>& container) const
    {
        typename sparse_matrix<int>::const_row_iterator it, et;

        container.clear();

        for (int k = 0; k != m; ++k) {
            std::tie(it, et) = ap.row(k);
            int v = 0;

            for (; it != et; ++it)
                v += A[it->value] * x[it->column];

            if (b[k] != v)
                container.emplace_back(k);
        }

        return length(container);
    }

    template<typename Xtype>
    double results(const Xtype& x,
                   const std::unique_ptr<Float[]>& original_costs,
                   const double cost_constant) const
    {
        bx_expects(is_valid_solution(x));

        auto value = static_cast<double>(cost_constant);

        for (int i{ 0 }, ei{ n }; i != ei; ++i)
            value += static_cast<double>(original_costs[i] * x[i]);

        return value;
    }

    //
    // Decrease influence of local preferences. 0 will completely reset the
    // preference values for the current row. > 0 will keep former decision in
    // mind.
    //
    void decrease_preference(sparse_matrix<int>::row_iterator begin,
                             sparse_matrix<int>::row_iterator end,
                             Float theta) noexcept
    {
        for (; begin != end; ++begin)
            P[begin->value] *= theta;
    }

    //
    // Compute the reduced costs and return the size of the newly R vector.
    //
    rc_size compute_reduced_costs(
      sparse_matrix<int>::row_iterator begin,
      sparse_matrix<int>::row_iterator end) noexcept
    {
        int r_size = 0;
        int c_size = 0;

        for (; begin != end; ++begin) {
            Float sum_a_pi = 0;
            Float sum_a_p = 0;

            auto ht = ap.column(begin->column);

            for (; std::get<0>(ht) != std::get<1>(ht); ++std::get<0>(ht)) {
                sum_a_pi += pi[std::get<0>(ht)->row];
                sum_a_p += P[std::get<0>(ht)->value];
            }

            R[r_size].id = r_size;
            R[r_size].value = c[begin->column] - sum_a_pi - sum_a_p;
            R[r_size].is_negative = A[begin->value] < 0;

            if (R[r_size].is_negative) {
                R[r_size].value = -R[r_size].value;
                ++c_size;
            }

            ++r_size;
        }

        return { r_size, c_size };
    }

    int select_variables(const rc_size& sizes, int bk)
    {
        return std::min(bk + sizes.c_size, sizes.r_size) - 1;
    }

    template<typename Xtype>
    void affect_variables(Xtype& x,
                          sparse_matrix<int>::row_iterator it,
                          int k,
                          int selected,
                          int r_size,
                          const Float kappa,
                          const Float delta) noexcept
    {
        constexpr Float one{ 1 };
        constexpr Float two{ 2 };
        constexpr Float middle{ (two + one) / two };

        auto d = delta;

        if (selected < 0) {
            pi[k] += R[0].value / two;
            d += (kappa / (one - kappa)) * (R[0].value / two);

            for (int i = 0; i != r_size; ++i) {
                auto var = it + R[i].id;

                if (R[i].is_negative) {
                    x.set(var->column, true);
                    P[var->value] += d;
                } else {
                    x.set(var->column, false);
                    P[var->value] -= d;
                }
            }
        } else if (selected + 1 >= r_size) {
            pi[k] += R[selected].value * middle;
            d += (kappa / (one - kappa)) * (R[selected].value * middle);

            for (int i = 0; i != r_size; ++i) {
                auto var = it + R[i].id;

                if (R[i].is_negative) {
                    x.set(var->column, false);
                    P[var->value] -= d;
                } else {
                    x.set(var->column, true);
                    P[var->value] += d;
                }
            }
        } else {
            pi[k] += ((R[selected].value + R[selected + 1].value) / two);
            d += (kappa / (one - kappa)) *
                 (R[selected + 1].value - R[selected].value);

            int i = 0;
            for (; i <= selected; ++i) {
                auto var = it + R[i].id;

                if (R[i].is_negative) {
                    x.set(var->column, false);
                    P[var->value] -= d;
                } else {
                    x.set(var->column, true);
                    P[var->value] += d;
                }
            }

            for (; i != r_size; ++i) {
                auto var = it + R[i].id;

                if (R[i].is_negative) {
                    x.set(var->column, true);
                    P[var->value] += d;
                } else {
                    x.set(var->column, false);
                    P[var->value] -= d;
                }
            }
        }

        bx_expects(is_valid_constraint(k, x));
    }

    template<typename Xtype, typename Iterator>
    void push_and_compute_update_row(Xtype& x,
                                     Iterator first,
                                     Iterator last,
                                     Float kappa,
                                     Float delta,
                                     Float theta,
                                     Float obj_amp)
    {
        for (; first != last; ++first) {
            auto k = constraint(first);

            const auto it = ap.row(k);
            decrease_preference(std::get<0>(it), std::get<1>(it), theta);

            const auto sizes =
              compute_reduced_costs(std::get<0>(it), std::get<1>(it));

            //
            // Before sort and select variables, we apply the push method: for
            // each reduces cost, we had the cost multiply with an objective
            // amplifier.
            //

            for (int i = 0; i != sizes.r_size; ++i)
                R[i].value += obj_amp * c[(std::get<0>(it) + R[i].id)->column];

            calculator_sort(R.get(), R.get() + sizes.r_size, rng, Mode());
            int selected = select_variables(sizes, b[k]);

            affect_variables(
              x, std::get<0>(it), k, selected, sizes.r_size, kappa, delta);
        }
    }

    template<typename Xtype, typename Iterator>
    void compute_update_row(Xtype& x,
                            Iterator first,
                            Iterator last,
                            Float kappa,
                            Float delta,
                            Float theta)
    {
        for (; first != last; ++first) {
            auto k = constraint(first);

            const auto it = ap.row(k);
            decrease_preference(std::get<0>(it), std::get<1>(it), theta);

            const auto sizes =
              compute_reduced_costs(std::get<0>(it), std::get<1>(it));

            calculator_sort(R.get(), R.get() + sizes.r_size, rng, Mode());
            int selected = select_variables(sizes, b[k]);

            affect_variables(
              x, std::get<0>(it), k, selected, sizes.r_size, kappa, delta);
        }
    }
};

template<typename Solver,
         typename Float,
         typename Mode,
         typename Order,
         typename Random>
static result
solve_or_optimize(const context_ptr& ctx,
                  const problem& pb,
                  bool is_optimization)
{
    return is_optimization
             ? optimize_problem<Solver, Float, Mode, Order, Random>(ctx, pb)
             : solve_problem<Solver, Float, Mode, Order, Random>(ctx, pb);
}

template<typename Float, typename Mode, typename Random>
static result
select_order(const context_ptr& ctx, const problem& pb, bool is_optimization)
{
    const auto c = static_cast<int>(ctx->parameters.order);

    if (c == 0)
        return solve_or_optimize<
          solver_equalities_101coeff<Float, Mode, Random>,
          Float,
          Mode,
          constraint_sel<Float, Random, 0>,
          Random>(ctx, pb, is_optimization);
    else if (c == 1)
        return solve_or_optimize<
          solver_equalities_101coeff<Float, Mode, Random>,
          Float,
          Mode,
          constraint_sel<Float, Random, 1>,
          Random>(ctx, pb, is_optimization);
    else if (c == 2)
        return solve_or_optimize<
          solver_equalities_101coeff<Float, Mode, Random>,
          Float,
          Mode,
          constraint_sel<Float, Random, 2>,
          Random>(ctx, pb, is_optimization);
    else if (c == 3)
        return solve_or_optimize<
          solver_equalities_101coeff<Float, Mode, Random>,
          Float,
          Mode,
          constraint_sel<Float, Random, 3>,
          Random>(ctx, pb, is_optimization);
    else
        return solve_or_optimize<
          solver_equalities_101coeff<Float, Mode, Random>,
          Float,
          Mode,
          constraint_sel<Float, Random, 4>,
          Random>(ctx, pb, is_optimization);
}

template<typename Float, typename Mode>
static result
select_random(const context_ptr& ctx, const problem& pb, bool is_optimization)
{
    return select_order<Float, Mode, std::default_random_engine>(
      ctx, pb, is_optimization);
}

template<typename Float>
static result
select_mode(const context_ptr& ctx, const problem& pb, bool is_optimization)
{
    const auto m = static_cast<int>(pb.type);

    return m == 0
             ? select_random<Float, mode_sel<0>>(ctx, pb, is_optimization)
             : select_random<Float, mode_sel<1>>(ctx, pb, is_optimization);
}

static result
select_float(const context_ptr& ctx, const problem& pb, bool is_optimization)
{
    const auto f = static_cast<int>(ctx->parameters.float_type);

    if (f == 0)
        return select_mode<float_sel<0>>(ctx, pb, is_optimization);
    else if (f == 1)
        return select_mode<float_sel<1>>(ctx, pb, is_optimization);
    else
        return select_mode<float_sel<2>>(ctx, pb, is_optimization);
}

result
solve_equalities_101(const context_ptr& ctx, const problem& pb)
{
    info(ctx, "  - solve_equalities_101\n");
    return select_float(ctx, pb, false);
}

result
optimize_equalities_101(const context_ptr& ctx, const problem& pb)
{
    info(ctx, "  - solve_equalities_101\n");
    return select_float(ctx, pb, true);
}

} // namespace itm
} // namespace baryonyx
