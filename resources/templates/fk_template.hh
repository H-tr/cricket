#pragma once

#include <vamp/vector.hh>
#include <vamp/vector/math.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>

// NOLINTBEGIN(*-magic-numbers)
namespace vamp::robots
{
struct {{name}}
{
    static constexpr char* name = "{{lower(name)}}";
    static constexpr std::size_t dimension = {{n_q}};
    static constexpr std::size_t n_spheres = {{n_spheres}};
    static constexpr float min_radius = {{min_radius}};
    static constexpr float max_radius = {{max_radius}};
    static constexpr std::size_t resolution = {{resolution}};

    static constexpr std::array<std::string_view, dimension> joint_names = {"{{join(joint_names, "\", \"")}}"};
    static constexpr char* end_effector = "{{end_effector}}";

    using Configuration = FloatVector<dimension>;
    using ConfigurationArray = std::array<FloatT, dimension>;

    struct alignas(FloatVectorAlignment) ConfigurationBuffer
        : std::array<float, Configuration::num_scalars_rounded>
    {
    };

    template <std::size_t rake>
    using ConfigurationBlock = FloatVector<rake, dimension>;

    template <std::size_t rake>
    struct Spheres
    {
        FloatVector<rake, n_spheres> x;
        FloatVector<rake, n_spheres> y;
        FloatVector<rake, n_spheres> z;
        FloatVector<rake, n_spheres> r;
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_m{
        {{join(bound_range, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_a{
        {{join(bound_lower, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> d_m{
        {{join(bound_descale, ", ")}}
    };

    static inline void scale_configuration(Configuration& q) noexcept
    {
        q = q * Configuration(s_m) + Configuration(s_a);
    }

    static inline void descale_configuration(Configuration& q) noexcept
    {
        q = (q - Configuration(s_a)) * Configuration(d_m);
    }

    template <std::size_t rake>
    static inline void scale_configuration_block(ConfigurationBlock<rake> &q) noexcept
    {
        {% for index in range(n_q) -%}
        q[{{index}}] = {{ at(bound_lower, index) }} + (q[{{index}}] * {{ at(bound_range, index) }});
        {%- endfor %}
    }

    template <std::size_t rake>
    static inline void descale_configuration_block(ConfigurationBlock<rake> & q) noexcept
    {
        {% for index in range(n_q) -%}
        q[{{index}}] = {{ at(bound_descale, index) }} * (q[{{index}}] - {{ at(bound_lower, index) }});
        {%- endfor %}
    }

    inline static auto space_measure() noexcept -> float
    {
        return {{measure}};
    }

    template <std::size_t rake>
    static inline void sphere_fk(const ConfigurationBlock<rake> &x, Spheres<rake> &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{spherefk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{spherefk_code_output}}> y;

        {{spherefk_code}}

        for (auto i = 0U; i < {{n_spheres}}; ++i)
        {
            out.x[i] = y[i * 4 + 0];
            out.y[i] = y[i * 4 + 1];
            out.z[i] = y[i * 4 + 2];
            out.r[i] = y[i * 4 + 3];
        }
    }

{% if exists("spherefk_with_cov_code") %}
    // Number of joint columns with Gaussian uncertainty, and the size
    // of the upper-triangular Σ_q passed to ``sphere_fk_with_cov``.
    static constexpr std::size_t n_uncertainty_cols = {{n_uncertainty_cols}};
    static constexpr std::size_t n_sigma_q = {{n_sigma_q}};

    // Per-sphere centre + symmetric 3×3 position covariance Σ_r^s in
    // the SoA SIMD layout that mirrors ``Spheres<rake>``.  The Σ
    // entries are the upper triangle row-major: σ_xx σ_xy σ_xz σ_yy σ_yz σ_zz.
    template <std::size_t rake>
    struct SpheresWithCov
    {
        FloatVector<rake, n_spheres> x;
        FloatVector<rake, n_spheres> y;
        FloatVector<rake, n_spheres> z;
        FloatVector<rake, n_spheres> r;
        FloatVector<rake, n_spheres> sigma_xx;
        FloatVector<rake, n_spheres> sigma_xy;
        FloatVector<rake, n_spheres> sigma_xz;
        FloatVector<rake, n_spheres> sigma_yy;
        FloatVector<rake, n_spheres> sigma_yz;
        FloatVector<rake, n_spheres> sigma_zz;
    };

    // Fused FK + base-uncertainty propagation.  Output per sphere s:
    //   (c_s ∈ R^3,  r_s,  Σ_r^s ∈ Sym3_+)
    // where Σ_r^s = J_s · Σ_q · J_s^T + r_s² · I and J_s is the analytic
    // Jacobian of c_s with respect to the configured uncertainty columns.
    // The body kernel r_s² · I is folded in at codegen time.
    //
    // Inputs:
    //   q_block       — rake-packed configurations (dimension joints per lane)
    //   sigma_q_upper — rake-packed Σ_q upper triangle (n_sigma_q floats per
    //                   lane; row-major).  Each lane may carry a distinct
    //                   covariance, matching the per-waypoint Σ_b(τ_k) shape
    //                   produced by an LQG-MP recursion.
    template <std::size_t rake>
    static inline void sphere_fk_with_cov(
        const ConfigurationBlock<rake> &q_block,
        const FloatVector<rake, n_sigma_q> &sigma_q_upper,
        SpheresWithCov<rake> &out) noexcept
    {
        // The codegen-emitted body references inputs through a single
        // flat ``x[i]`` array.  Lay out (q first, Σ_q after) into one
        // backing array so the indices match what CppADCodeGen wrote.
        std::array<FloatVector<rake, 1>, dimension + n_sigma_q> x;
        for (auto i = 0U; i < dimension; ++i)
        {
            x[i] = q_block[i];
        }
        for (auto i = 0U; i < n_sigma_q; ++i)
        {
            x[dimension + i] = sigma_q_upper[i];
        }

        std::array<FloatVector<rake, 1>, {{spherefk_with_cov_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{spherefk_with_cov_code_output}}> y;

        {{spherefk_with_cov_code}}

        for (auto i = 0U; i < {{n_spheres}}; ++i)
        {
            out.x[i]        = y[i * 10 + 0];
            out.y[i]        = y[i * 10 + 1];
            out.z[i]        = y[i * 10 + 2];
            out.r[i]        = y[i * 10 + 3];
            out.sigma_xx[i] = y[i * 10 + 4];
            out.sigma_xy[i] = y[i * 10 + 5];
            out.sigma_xz[i] = y[i * 10 + 6];
            out.sigma_yy[i] = y[i * 10 + 7];
            out.sigma_yz[i] = y[i * 10 + 8];
            out.sigma_zz[i] = y[i * 10 + 9];
        }
    }
{% endif %}

    using Debug = std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<std::size_t, std::size_t>>>;

    template <std::size_t rake>
        static inline auto fkcc_debug(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept -> Debug
    {
        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}

        Debug output;

        {% for i in range(n_spheres) %}
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[{{ i * 4 + 0 }}],
                y[{{ i * 4 + 1 }}],
                y[{{ i * 4 + 2 }}],
                y[{{ i * 4 + 3 }}]));
        {% endfor %}

        {% for i in range(length(allowed_link_pairs)) %}
        {% set pair = at(allowed_link_pairs, i) %}
        {% set link_1_index = at(pair, 0) %}
        {% set link_2_index = at(pair, 1) %}
        {% set link_1_spheres = at(per_link_spheres, link_1_index) %}
        {% set link_2_spheres = at(per_link_spheres, link_2_index) %}

        {% for j in range(length(link_1_spheres)) %}
        {% for k in range(length(link_2_spheres)) %}

        {% set sphere_1_loc = at(link_1_spheres, j) %}
        {% set sphere_2_loc = at(link_2_spheres, k) %}

        if (sphere_sphere_self_collision<decltype(x[0])>(y[{{ sphere_1_loc * 4 + 0}} ],
                                                         y[{{ sphere_1_loc * 4 + 1}} ],
                                                         y[{{ sphere_1_loc * 4 + 2}} ],
                                                         y[{{ sphere_1_loc * 4 + 3}} ],
                                                         y[{{ sphere_2_loc * 4 + 0}} ],
                                                         y[{{ sphere_2_loc * 4 + 1}} ],
                                                         y[{{ sphere_2_loc * 4 + 2}} ],
                                                         y[{{ sphere_2_loc * 4 + 3}} ]))
        {
            output.second.emplace_back({{ sphere_1_loc }}, {{ sphere_2_loc }});
        }

        {% endfor %}
        {% endfor %}
        {% endfor %}

        return output;
    }

    template <std::size_t rake>
        static inline bool fkcc(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept
    {
        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}
        {% include "ccfk" %}

        return true;
    }

    template <std::size_t rake>
    static inline bool fkcc_attach(
        const vamp::collision::Environment<FloatVector<rake>> &environment,
        const ConfigurationBlock<rake> &x) noexcept
    {
        std::array<FloatVector<rake, 1>, {{ccfkee_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfkee_code_output}}> y;

        {{ccfkee_code}}
        {% include "ccfk" %}

        // attaching at {{ end_effector }}
        set_attachment_pose(environment, to_isometry(&y[{{ccfkee_code_output - 12}}]));

        //
        // attachment vs. environment collisions
        //
        if (attachment_environment_collision(environment))
        {
            return false;
        }

        //
        // attachment vs. robot collisions
        //

        {% for i in range(length(end_effector_collisions)) %}
        {% set link_index = at(end_effector_collisions, i) %}
        {% set link_bs = at(bounding_sphere_index, link_index) %}
        {% set link_spheres = at(per_link_spheres, link_index) %}

        // Attachment vs. {{ at(link_names, link_index )}}
        if (attachment_sphere_collision<decltype(x[0])>(environment,
                                                        y[{{(n_spheres + link_bs) * 4 + 0}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 1}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 2}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 3}}]))
        {
            {% for j in range(length(link_spheres)) %}
            {% set sphere_index = at(link_spheres, j) %}
            if (attachment_sphere_collision<decltype(x[0])>(environment,
                                                            y[{{sphere_index * 4 + 0}}],
                                                            y[{{sphere_index * 4 + 1}}],
                                                            y[{{sphere_index * 4 + 2}}],
                                                            y[{{sphere_index * 4 + 3}}]))
            {
                return false;
            }
            {% endfor %}
        }
        {% endfor %}

        return true;
    }

    static inline auto eefk(const std::array<float, {{n_q}}> &x) noexcept -> Eigen::Isometry3f
    {
        std::array<float, {{eefk_code_vars}}> v;
        std::array<float, {{eefk_code_output}}> y;

        {{eefk_code}}

        return to_isometry(y.data());
    }
};
}

// NOLINTEND(*-magic-numbers)
